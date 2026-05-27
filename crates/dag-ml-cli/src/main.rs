use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command as ProcessCommand, Stdio};
use std::sync::{
    mpsc::{self, Receiver, RecvTimeoutError},
    Mutex,
};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use dag_ml_core::{
    build_execution_bundle, build_execution_bundle_with_prediction_contracts, build_execution_plan,
    build_openlineage_run_event_from_package_files, build_prediction_cache_payload,
    build_prediction_cache_record, build_research_provenance_package, compile_pipeline_dsl,
    compile_pipeline_dsl_with_generation, oof_campaign_fingerprint,
    regression_report_to_candidate_score, score_regression_aggregated_block,
    score_regression_prediction_block, select_candidate, select_candidate_groups,
    validate_oof_campaign, validate_research_provenance_package_files, AggregatedPredictionBlock,
    ArtifactId, BundleId, BundlePredictionCachePayload, BundlePredictionCachePayloadSet,
    BundlePredictionCacheRecord, BundlePredictionRequirement, BundleReplayExecution, CampaignSpec,
    CandidateScore, ColumnarPredictionCacheStore, ControllerId, ControllerManifest,
    ControllerRegistry, DagMlError, DataRequestPartition, ExecutionBundle,
    ExternalDataPlanEnvelope, FileArtifactManifestStore, FileArtifactPayloadStore,
    FilePredictionCacheStore, GraphSpec, HandleKind, HandleRef, InMemoryArtifactStore,
    InMemoryDataProvider, LineageId, LineageRecord, MetricObjective, NodeId, NodeResult, NodeTask,
    OofCampaign, ParallelScheduler, Phase, PipelineDslSpec, PredictionBlock, PredictionLevel,
    PredictionPartition, RefitArtifactRecord, RegressionMetricKind, RegressionMetricReport,
    RegressionTargetBlock, ReplayPhaseRequest, ResearchProvenancePackage, RunContext, RunId,
    RuntimeArtifactStore, RuntimeController, RuntimeControllerRegistry, RuntimeDataProvider,
    RuntimePredictionCacheStore, SampleId, SelectionDecision, SelectionMetric, SelectionPolicy,
    SequentialScheduler, VariantId,
};
use serde::{Deserialize, Serialize};

const DEFAULT_PROCESS_TIMEOUT_MS: u64 = 30_000;
const PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION: u32 = 1;
const PROCESS_ADAPTER_PROTOCOL: &str = "dag-ml-process-adapter";
const PROCESS_ADAPTER_MODE_ONE_SHOT: &str = "one_shot";
const PROCESS_ADAPTER_MODE_JSONL: &str = "jsonl";
const PROCESS_ADAPTER_CAP_NODE_TASK_JSON: &str = "node_task_json_v1";
const PROCESS_ADAPTER_CAP_NODE_RESULT_JSON: &str = "node_result_json_v1";
const PROCESS_ADAPTER_CAP_CONTROL_FRAMES: &str = "control_frames_v1";
const PROCESS_ADAPTER_CAP_PARALLEL_INVOCATION: &str = "parallel_invocation_v1";
const PROCESS_ADAPTER_CAP_PERSISTENT_WORKERS: &str = "persistent_workers";
const PROCESS_ADAPTER_CAP_WORKER_ENV: &str = "worker_env";
const PROCESS_ADAPTER_FRAME_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliScheduler {
    Sequential,
    Parallel,
}

impl CliScheduler {
    fn label(self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliPredictionBlockKind {
    Sample,
    Aggregated,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum CliRegressionMetricKind {
    Mse,
    Rmse,
    Mae,
    R2,
}

impl From<CliRegressionMetricKind> for RegressionMetricKind {
    fn from(value: CliRegressionMetricKind) -> Self {
        match value {
            CliRegressionMetricKind::Mse => Self::Mse,
            CliRegressionMetricKind::Rmse => Self::Rmse,
            CliRegressionMetricKind::Mae => Self::Mae,
            CliRegressionMetricKind::R2 => Self::R2,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SchedulerConfig {
    scheduler: CliScheduler,
    workers: usize,
}

impl SchedulerConfig {
    fn new(scheduler: CliScheduler, workers: usize) -> Result<Self> {
        if workers == 0 {
            bail!("--scheduler-workers must be at least 1");
        }
        Ok(Self { scheduler, workers })
    }
}

fn execute_campaign_phase_with_scheduler(
    scheduler: SchedulerConfig,
    plan: &dag_ml_core::ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    data_provider: &dyn RuntimeDataProvider,
    ctx: &mut RunContext,
    phase: Phase,
) -> Result<Vec<NodeResult>> {
    match scheduler.scheduler {
        CliScheduler::Sequential => Ok(SequentialScheduler
            .execute_campaign_phase_with_data_provider(
                plan,
                controllers,
                data_provider,
                ctx,
                phase,
            )?),
        CliScheduler::Parallel => Ok(ParallelScheduler::new(scheduler.workers)?
            .execute_campaign_phase_with_data_provider(
                plan,
                controllers,
                data_provider,
                ctx,
                phase,
            )?),
    }
}

fn execute_campaign_phase_with_artifact_store_and_scheduler(
    scheduler: SchedulerConfig,
    plan: &dag_ml_core::ExecutionPlan,
    controllers: &RuntimeControllerRegistry,
    data_provider: &dyn RuntimeDataProvider,
    artifact_store: &mut InMemoryArtifactStore,
    ctx: &mut RunContext,
    phase: Phase,
) -> Result<Vec<NodeResult>> {
    match scheduler.scheduler {
        CliScheduler::Sequential => Ok(SequentialScheduler
            .execute_campaign_phase_with_data_provider_and_artifact_store(
                plan,
                controllers,
                data_provider,
                artifact_store,
                ctx,
                phase,
            )?),
        CliScheduler::Parallel => Ok(ParallelScheduler::new(scheduler.workers)?
            .execute_campaign_phase_with_data_provider_and_artifact_store(
                plan,
                controllers,
                data_provider,
                artifact_store,
                ctx,
                phase,
            )?),
    }
}

fn execute_bundle_replay_with_scheduler(
    scheduler: SchedulerConfig,
    replay: BundleReplayExecution<'_>,
    ctx: &mut RunContext,
) -> Result<Vec<NodeResult>> {
    match scheduler.scheduler {
        CliScheduler::Sequential => Ok(SequentialScheduler.execute_bundle_replay(replay, ctx)?),
        CliScheduler::Parallel => {
            Ok(ParallelScheduler::new(scheduler.workers)?.execute_bundle_replay(replay, ctx)?)
        }
    }
}

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
    CompilePipelineDsl {
        #[arg(long)]
        dsl: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        artifact: bool,
    },
    BuildPipelineDslPlan {
        #[arg(long)]
        dsl: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long, default_value = "plan:cli.dsl")]
        plan_id: String,
        #[arg(long)]
        output: Option<PathBuf>,
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
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
        #[arg(long, default_value_t = 1)]
        process_workers: usize,
        #[arg(long, default_value_t = DEFAULT_PROCESS_TIMEOUT_MS)]
        process_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        process_retries: usize,
        #[arg(long, default_value = "plan:cli.process")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
    ScoreRegression {
        #[arg(long, value_enum)]
        prediction_block: CliPredictionBlockKind,
        #[arg(long)]
        predictions: PathBuf,
        #[arg(long)]
        targets: PathBuf,
        #[arg(long, value_enum, required = true)]
        metric: Vec<CliRegressionMetricKind>,
        #[arg(long)]
        candidate_id: Option<String>,
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
        #[arg(long)]
        lineage_output: Option<PathBuf>,
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
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
        #[arg(long, default_value_t = 1)]
        process_workers: usize,
        #[arg(long, default_value_t = DEFAULT_PROCESS_TIMEOUT_MS)]
        process_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        process_retries: usize,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        lineage_output: Option<PathBuf>,
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
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
        #[arg(long, default_value_t = 1)]
        process_workers: usize,
        #[arg(long, default_value_t = DEFAULT_PROCESS_TIMEOUT_MS)]
        process_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        process_retries: usize,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        lineage_output: Option<PathBuf>,
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
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
        #[arg(long, default_value_t = 1)]
        process_workers: usize,
        #[arg(long, default_value_t = DEFAULT_PROCESS_TIMEOUT_MS)]
        process_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        process_retries: usize,
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
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
    },
    RunProcessDslCvRefitBundle {
        #[arg(long)]
        dsl: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long)]
        envelope: PathBuf,
        #[arg(long)]
        adapter: PathBuf,
        #[arg(long)]
        persistent: bool,
        #[arg(long, default_value_t = 1)]
        process_workers: usize,
        #[arg(long, default_value_t = DEFAULT_PROCESS_TIMEOUT_MS)]
        process_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        process_retries: usize,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long)]
        lineage_output: Option<PathBuf>,
        #[arg(long)]
        prediction_cache_output: Option<PathBuf>,
        #[arg(long, default_value = "bundle:cli.process.dsl.cv.refit")]
        bundle_id: String,
        #[arg(long)]
        variant_id: Option<String>,
        #[arg(long)]
        selections: Option<PathBuf>,
        #[arg(long, default_value = "plan:cli.process.dsl.cv.refit")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process.dsl.cv.refit")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
    },
    RunProcessDslCvRefitReplay {
        #[arg(long)]
        dsl: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long)]
        envelope: PathBuf,
        #[arg(long)]
        adapter: PathBuf,
        #[arg(long, default_value_t = 1)]
        process_workers: usize,
        #[arg(long, default_value_t = DEFAULT_PROCESS_TIMEOUT_MS)]
        process_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        process_retries: usize,
        #[arg(long, default_value = "bundle:cli.process.dsl.cv.refit.replay")]
        bundle_id: String,
        #[arg(long)]
        variant_id: Option<String>,
        #[arg(long)]
        selections: Option<PathBuf>,
        #[arg(long, default_value = "plan:cli.process.dsl.cv.refit.replay")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process.dsl.cv.refit.replay")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
        #[arg(long, default_value_t = 1)]
        process_workers: usize,
        #[arg(long, default_value_t = DEFAULT_PROCESS_TIMEOUT_MS)]
        process_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        process_retries: usize,
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
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
        #[arg(long)]
        artifact_manifest: Option<PathBuf>,
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
    ExportArtifactManifest {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    ValidateArtifactManifest {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        manifest_dir: PathBuf,
    },
    ExportArtifactPayloadStore {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        source_dir: PathBuf,
        #[arg(long)]
        output_dir: PathBuf,
    },
    ValidateArtifactPayloadStore {
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        store_dir: PathBuf,
    },
    ExportResearchProvenance {
        #[arg(long)]
        graph: PathBuf,
        #[arg(long)]
        campaign: PathBuf,
        #[arg(long)]
        controllers: PathBuf,
        #[arg(long)]
        bundle: PathBuf,
        #[arg(long)]
        lineage: Option<PathBuf>,
        #[arg(long)]
        artifact_manifest: Option<PathBuf>,
        #[arg(long)]
        prediction_cache_store: Option<PathBuf>,
        #[arg(long = "envelope")]
        envelopes: Vec<String>,
        #[arg(long)]
        output_dir: PathBuf,
        #[arg(long, default_value = "plan:cli.bundle")]
        plan_id: String,
    },
    ValidateResearchProvenance {
        #[arg(long)]
        input_dir: PathBuf,
    },
    ExportOpenLineage {
        #[arg(long)]
        input_dir: PathBuf,
        #[arg(long)]
        event_time: String,
        #[arg(long, default_value = "dag-ml")]
        namespace: String,
        #[arg(long)]
        output: Option<PathBuf>,
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
        #[arg(long)]
        artifact_payload_store: Option<PathBuf>,
        #[arg(long = "envelope")]
        envelopes: Vec<String>,
        #[arg(long, default_value = "plan:cli.bundle")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.replay")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
        #[arg(long, default_value_t = 1)]
        process_workers: usize,
        #[arg(long, default_value_t = DEFAULT_PROCESS_TIMEOUT_MS)]
        process_timeout_ms: u64,
        #[arg(long, default_value_t = 0)]
        process_retries: usize,
        #[arg(long, default_value = "plan:cli.bundle")]
        plan_id: String,
        #[arg(long, default_value = "run:cli.process.replay")]
        run_id: String,
        #[arg(long, default_value_t = 12345)]
        root_seed: u64,
        #[arg(long, value_enum, default_value = "sequential")]
        scheduler: CliScheduler,
        #[arg(long, default_value_t = 1)]
        scheduler_workers: usize,
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
        Command::CompilePipelineDsl {
            dsl,
            output,
            artifact,
        } => {
            let spec: PipelineDslSpec = read_json(&dsl, "pipeline DSL")?;
            if artifact {
                let compiled = compile_pipeline_dsl_with_generation(&spec).with_context(|| {
                    format!("failed to compile pipeline DSL at {}", dsl.display())
                })?;
                emit_json(output.as_ref(), &compiled, "compiled pipeline DSL artifact")?;
            } else {
                let graph = compile_pipeline_dsl(&spec).with_context(|| {
                    format!("failed to compile pipeline DSL at {}", dsl.display())
                })?;
                emit_json(output.as_ref(), &graph, "compiled graph")?;
            }
        }
        Command::BuildPipelineDslPlan {
            dsl,
            controllers,
            plan_id,
            output,
        } => {
            let plan = build_plan_from_dsl_path(&dsl, &controllers, plan_id)?;
            emit_json(output.as_ref(), &plan, "pipeline DSL execution plan")?;
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
            scheduler,
            scheduler_workers,
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
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let results = execute_campaign_phase_with_scheduler(
                scheduler,
                &plan,
                &runtime_controllers,
                &data_provider,
                &mut ctx,
                Phase::FitCv,
            )
            .with_context(|| "mock campaign execution failed")?;
            println!(
                "mock campaign run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s), scheduler={}, scheduler worker(s)={}",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len(),
                scheduler.scheduler.label(),
                scheduler.workers
            );
        }
        Command::RunProcessCampaign {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            persistent,
            process_workers,
            process_timeout_ms,
            process_retries,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
            let process_config = process_adapter_runtime_config(
                process_workers,
                process_timeout_ms,
                process_retries,
            )?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let runtime_controllers = process_runtime_controllers_for_mode(
                &plan,
                adapter,
                persistent,
                process_config,
                scheduler,
            )?;
            let mut ctx = RunContext::new(RunId::new(run_id)?, Some(root_seed));
            let results = execute_campaign_phase_with_scheduler(
                scheduler,
                &plan,
                &runtime_controllers,
                &data_provider,
                &mut ctx,
                Phase::FitCv,
            )
            .with_context(|| "process campaign execution failed")?;
            println!(
                "process campaign run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s), scheduler={}, scheduler worker(s)={}, configured process worker(s)={}, observed process worker(s)={}",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len(),
                scheduler.scheduler.label(),
                scheduler.workers,
                configured_persistent_process_workers(persistent, process_workers),
                observed_persistent_process_worker_count(persistent, &ctx)
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
        Command::ScoreRegression {
            prediction_block,
            predictions,
            targets,
            metric,
            candidate_id,
            output,
        } => {
            let targets: RegressionTargetBlock = read_json(&targets, "regression target block")?;
            let metrics = metric.into_iter().map(Into::into).collect::<Vec<_>>();
            let report = match prediction_block {
                CliPredictionBlockKind::Sample => {
                    let predictions: PredictionBlock =
                        read_json(&predictions, "sample prediction block")?;
                    score_regression_prediction_block(&predictions, &targets, &metrics)
                }
                CliPredictionBlockKind::Aggregated => {
                    let predictions: AggregatedPredictionBlock =
                        read_json(&predictions, "aggregated prediction block")?;
                    score_regression_aggregated_block(&predictions, &targets, &metrics)
                }
            }
            .with_context(|| "regression scoring failed")?;
            let candidate_score = candidate_id
                .map(|candidate_id| {
                    regression_report_to_candidate_score(candidate_id, report.clone())
                })
                .transpose()
                .with_context(|| "failed to convert regression report to candidate score")?;
            emit_json(
                output.as_ref(),
                &RegressionScoreCliOutput {
                    report,
                    candidate_score,
                },
                "regression score",
            )?;
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
            lineage_output,
            bundle_id,
            variant_id,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
            let runtime_controllers = mock_runtime_controllers_with_refit_artifacts(&plan)?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let captured = build_bundle_from_captured_refit(CapturedRefitBundleInput {
                plan: &plan,
                data_provider: &data_provider,
                runtime_controllers: &runtime_controllers,
                bundle_id,
                variant_id,
                selections: BTreeMap::new(),
                run_id,
                root_seed,
                scheduler,
            })
            .with_context(|| "mock refit bundle capture failed")?;
            emit_json(output.as_ref(), &captured.bundle, "execution bundle")?;
            emit_json(
                lineage_output.as_ref(),
                &captured.lineage_records,
                "lineage records",
            )?;
        }
        Command::RunProcessRefitBundle {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            persistent,
            process_workers,
            process_timeout_ms,
            process_retries,
            output,
            lineage_output,
            bundle_id,
            variant_id,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
            let process_config = process_adapter_runtime_config(
                process_workers,
                process_timeout_ms,
                process_retries,
            )?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let runtime_controllers = process_runtime_controllers_for_mode(
                &plan,
                adapter,
                persistent,
                process_config,
                scheduler,
            )?;
            let captured = build_bundle_from_captured_refit(CapturedRefitBundleInput {
                plan: &plan,
                data_provider: &data_provider,
                runtime_controllers: &runtime_controllers,
                bundle_id,
                variant_id,
                selections: BTreeMap::new(),
                run_id,
                root_seed,
                scheduler,
            })
            .with_context(|| "process refit bundle capture failed")?;
            emit_json(output.as_ref(), &captured.bundle, "execution bundle")?;
            emit_json(
                lineage_output.as_ref(),
                &captured.lineage_records,
                "lineage records",
            )?;
        }
        Command::RunProcessCvRefitBundle {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            persistent,
            process_workers,
            process_timeout_ms,
            process_retries,
            output,
            lineage_output,
            prediction_cache_output,
            bundle_id,
            variant_id,
            selections,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
            let process_config = process_adapter_runtime_config(
                process_workers,
                process_timeout_ms,
                process_retries,
            )?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let runtime_controllers = process_runtime_controllers_for_mode(
                &plan,
                adapter,
                persistent,
                process_config,
                scheduler,
            )?;
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
                scheduler,
            })
            .with_context(|| "process CV+refit bundle capture failed")?;
            println!(
                "process cv refit bundle run: {} fit_cv result(s), {} OOF prediction block(s), {} refit result(s), {} captured artifact handle(s), {} prediction cache(s), scheduler={}, scheduler worker(s)={}, configured process worker(s)={}, observed process worker(s)={}",
                captured.fit_cv_result_count,
                captured.fit_cv_oof_prediction_block_count,
                captured.refit_result_count,
                captured.artifact_store.len(),
                captured.bundle.prediction_caches.len(),
                scheduler.scheduler.label(),
                scheduler.workers,
                configured_persistent_process_workers(persistent, process_workers),
                if persistent {
                    captured.observed_process_worker_count
                } else {
                    0
                }
            );
            emit_json(output.as_ref(), &captured.bundle, "execution bundle")?;
            emit_json(
                lineage_output.as_ref(),
                &captured.lineage_records,
                "lineage records",
            )?;
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
            process_workers,
            process_timeout_ms,
            process_retries,
            bundle_id,
            variant_id,
            selections,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope.clone())?;
            let process_config = process_adapter_runtime_config(
                process_workers,
                process_timeout_ms,
                process_retries,
            )?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let runtime_controllers =
                persistent_process_runtime_controllers(&plan, adapter, process_config, scheduler)?;
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
                scheduler,
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
            let replay_results = execute_bundle_replay_with_scheduler(
                scheduler,
                BundleReplayExecution {
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
                "process cv refit replay run: {} fit_cv result(s), {} OOF prediction block(s), {} refit result(s), {} replay result(s), {} replay prediction block(s), {} captured artifact handle(s), {} prediction cache(s), scheduler={}, scheduler worker(s)={}, configured process worker(s)={}, observed process worker(s)={}, replay observed process worker(s)={}",
                captured.fit_cv_result_count,
                captured.fit_cv_oof_prediction_block_count,
                captured.refit_result_count,
                replay_results.len(),
                replay_ctx.prediction_store.blocks().len(),
                captured.artifact_store.len(),
                captured.bundle.prediction_caches.len(),
                scheduler.scheduler.label(),
                scheduler.workers,
                process_workers,
                captured.observed_process_worker_count,
                observed_process_worker_count(&replay_ctx)
            );
        }
        Command::RunProcessDslCvRefitBundle {
            dsl,
            controllers,
            envelope,
            adapter,
            persistent,
            process_workers,
            process_timeout_ms,
            process_retries,
            output,
            lineage_output,
            prediction_cache_output,
            bundle_id,
            variant_id,
            selections,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
        } => {
            let plan = build_plan_from_dsl_path(&dsl, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope)?;
            let process_config = process_adapter_runtime_config(
                process_workers,
                process_timeout_ms,
                process_retries,
            )?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let runtime_controllers = process_runtime_controllers_for_mode(
                &plan,
                adapter,
                persistent,
                process_config,
                scheduler,
            )?;
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
                scheduler,
            })
            .with_context(|| "process DSL CV+refit bundle capture failed")?;
            println!(
                "process DSL cv refit bundle run: {} fit_cv result(s), {} OOF prediction block(s), {} refit result(s), {} captured artifact handle(s), {} prediction cache(s), scheduler={}, scheduler worker(s)={}, configured process worker(s)={}, observed process worker(s)={}",
                captured.fit_cv_result_count,
                captured.fit_cv_oof_prediction_block_count,
                captured.refit_result_count,
                captured.artifact_store.len(),
                captured.bundle.prediction_caches.len(),
                scheduler.scheduler.label(),
                scheduler.workers,
                configured_persistent_process_workers(persistent, process_workers),
                if persistent {
                    captured.observed_process_worker_count
                } else {
                    0
                }
            );
            emit_json(output.as_ref(), &captured.bundle, "execution bundle")?;
            emit_json(
                lineage_output.as_ref(),
                &captured.lineage_records,
                "lineage records",
            )?;
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
        Command::RunProcessDslCvRefitReplay {
            dsl,
            controllers,
            envelope,
            adapter,
            process_workers,
            process_timeout_ms,
            process_retries,
            bundle_id,
            variant_id,
            selections,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
        } => {
            let plan = build_plan_from_dsl_path(&dsl, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope.clone())?;
            let process_config = process_adapter_runtime_config(
                process_workers,
                process_timeout_ms,
                process_retries,
            )?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let runtime_controllers =
                persistent_process_runtime_controllers(&plan, adapter, process_config, scheduler)?;
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
                scheduler,
            })
            .with_context(|| "process DSL CV+refit capture before replay failed")?;
            let envelope_map = replay_envelope_map_for_bundle(&captured.bundle, &envelope);
            let replay_request = ReplayPhaseRequest {
                bundle_id: captured.bundle.bundle_id.clone(),
                phase: Phase::Predict,
                data_envelope_keys: envelope_map.keys().cloned().collect(),
            };
            let mut replay_ctx =
                RunContext::new(RunId::new(format!("{run_id}:predict"))?, Some(root_seed));
            let replay_results = execute_bundle_replay_with_scheduler(
                scheduler,
                BundleReplayExecution {
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
            .with_context(|| "process DSL replay after CV+refit capture failed")?;
            println!(
                "process DSL cv refit replay run: {} fit_cv result(s), {} OOF prediction block(s), {} refit result(s), {} replay result(s), {} replay prediction block(s), {} captured artifact handle(s), {} prediction cache(s), scheduler={}, scheduler worker(s)={}, configured process worker(s)={}, observed process worker(s)={}, replay observed process worker(s)={}",
                captured.fit_cv_result_count,
                captured.fit_cv_oof_prediction_block_count,
                captured.refit_result_count,
                replay_results.len(),
                replay_ctx.prediction_store.blocks().len(),
                captured.artifact_store.len(),
                captured.bundle.prediction_caches.len(),
                scheduler.scheduler.label(),
                scheduler.workers,
                process_workers,
                captured.observed_process_worker_count,
                observed_process_worker_count(&replay_ctx)
            );
        }
        Command::RunProcessRefitReplay {
            graph,
            campaign,
            controllers,
            envelope,
            adapter,
            process_workers,
            process_timeout_ms,
            process_retries,
            bundle_id,
            variant_id,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let envelope: ExternalDataPlanEnvelope =
                read_json(&envelope, "external data-plan envelope")?;
            let data_provider = data_provider_for_training_envelope(&plan, envelope.clone())?;
            let process_config = process_adapter_runtime_config(
                process_workers,
                process_timeout_ms,
                process_retries,
            )?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let runtime_controllers =
                persistent_process_runtime_controllers(&plan, adapter, process_config, scheduler)?;
            let captured = build_bundle_from_captured_refit(CapturedRefitBundleInput {
                plan: &plan,
                data_provider: &data_provider,
                runtime_controllers: &runtime_controllers,
                bundle_id,
                variant_id,
                selections: BTreeMap::new(),
                run_id: run_id.clone(),
                root_seed,
                scheduler,
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
            let replay_results = execute_bundle_replay_with_scheduler(
                scheduler,
                BundleReplayExecution {
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
                "process refit replay run: {} refit result(s), {} replay result(s), {} replay prediction block(s), {} captured artifact handle(s), scheduler={}, scheduler worker(s)={}, configured process worker(s)={}, observed process worker(s)={}, replay observed process worker(s)={}",
                captured.refit_result_count,
                replay_results.len(),
                replay_ctx.prediction_store.blocks().len(),
                captured.artifact_store.len(),
                scheduler.scheduler.label(),
                scheduler.workers,
                process_workers,
                captured.observed_process_worker_count,
                observed_process_worker_count(&replay_ctx)
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
            artifact_manifest,
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
            let file_artifact_manifest_store = artifact_manifest
                .as_ref()
                .map(|manifest_dir| validate_file_artifact_manifest_store(&bundle, manifest_dir))
                .transpose()?;
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
                "valid bundle: {}, selection(s)={}, artifact(s)={}, prediction requirement(s)={}, prediction cache(s)={}, prediction cache payload(s)={}, prediction cache store cache(s)={}, data requirement(s)={}, replay envelope(s)={}, artifact manifest entries={}",
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
                envelope_map.len(),
                file_artifact_manifest_store
                    .as_ref()
                    .map_or(0, |store| store.manifest().artifacts.len())
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
        Command::ExportArtifactManifest { bundle, output_dir } => {
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let manifest = FileArtifactManifestStore::write(&output_dir, &bundle)
                .with_context(|| "failed to export artifact manifest")?;
            println!(
                "wrote artifact manifest: bundle={}, artifact(s)={}, dir={}",
                manifest.bundle_id,
                manifest.artifacts.len(),
                output_dir.display()
            );
        }
        Command::ValidateArtifactManifest {
            bundle,
            manifest_dir,
        } => {
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let store = validate_file_artifact_manifest_store(&bundle, &manifest_dir)?;
            println!(
                "valid artifact manifest: bundle={}, artifact(s)={}, dir={}",
                store.manifest().bundle_id,
                store.manifest().artifacts.len(),
                manifest_dir.display()
            );
        }
        Command::ExportArtifactPayloadStore {
            bundle,
            source_dir,
            output_dir,
        } => {
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let store =
                FileArtifactPayloadStore::write_from_source(&output_dir, &source_dir, &bundle)
                    .with_context(|| "failed to export artifact payload store")?;
            println!(
                "wrote artifact payload store: bundle={}, artifact(s)={}, dir={}",
                store.manifest().bundle_id,
                store.payload_count(),
                output_dir.display()
            );
        }
        Command::ValidateArtifactPayloadStore { bundle, store_dir } => {
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let store = validate_file_artifact_payload_store(&bundle, &store_dir)?;
            println!(
                "valid artifact payload store: bundle={}, artifact(s)={}, dir={}",
                store.manifest().bundle_id,
                store.payload_count(),
                store_dir.display()
            );
        }
        Command::ExportResearchProvenance {
            graph,
            campaign,
            controllers,
            bundle,
            lineage,
            artifact_manifest,
            prediction_cache_store,
            envelopes,
            output_dir,
            plan_id,
        } => {
            let plan = build_plan_from_paths(&graph, &campaign, &controllers, plan_id)?;
            let bundle: ExecutionBundle = read_json(&bundle, "execution bundle")?;
            let lineage_records: Vec<LineageRecord> = lineage
                .as_ref()
                .map(|path| read_json(path, "lineage record list"))
                .transpose()?
                .unwrap_or_default();
            let envelope_map = read_replay_envelopes(&envelopes)?;
            let prediction_cache_store = prediction_cache_store
                .as_ref()
                .map(|store_dir| validate_file_prediction_cache_store(&bundle, store_dir))
                .transpose()?;
            let artifact_manifest_store = artifact_manifest
                .as_ref()
                .map(|manifest_dir| validate_file_artifact_manifest_store(&bundle, manifest_dir))
                .transpose()?;
            let package = build_research_provenance_package(
                &plan,
                &bundle,
                &lineage_records,
                &envelope_map,
                prediction_cache_store
                    .as_ref()
                    .map(|store| store.manifest()),
                artifact_manifest_store
                    .as_ref()
                    .map(|store| store.manifest()),
            )
            .with_context(|| "failed to build research provenance export")?;

            write_research_provenance_package(&output_dir, &package)?;
            println!(
                "wrote research provenance export: bundle={}, lineage record(s)={}, data envelope(s)={}, prediction cache manifest={}, artifact manifest={}, file(s)={}, dir={}",
                bundle.bundle_id,
                lineage_records.len(),
                envelope_map.len(),
                prediction_cache_store.is_some(),
                artifact_manifest_store.is_some(),
                package.files.len(),
                output_dir.display()
            );
        }
        Command::ValidateResearchProvenance { input_dir } => {
            let files = read_research_provenance_package_dir(&input_dir)?;
            let validation = validate_research_provenance_package_files(&files)
                .with_context(|| "failed to validate research provenance package")?;
            println!(
                "valid research provenance package: bundle={}, plan={}, file(s)={}, checksummed file(s)={}, lineage record(s)={}, data envelope(s)={}, prediction cache manifest={}, artifact manifest={}, dir={}",
                validation.bundle_id,
                validation.plan_id,
                validation.file_count,
                validation.checksummed_file_count,
                validation.lineage_record_count,
                validation.data_envelope_count,
                validation.has_prediction_cache_manifest,
                validation.has_artifact_manifest,
                input_dir.display()
            );
        }
        Command::ExportOpenLineage {
            input_dir,
            event_time,
            namespace,
            output,
        } => {
            let files = read_research_provenance_package_dir(&input_dir)?;
            let event = build_openlineage_run_event_from_package_files(
                &files,
                namespace.as_str(),
                event_time.as_str(),
            )
            .with_context(|| "failed to build OpenLineage run event")?;
            emit_json(output.as_ref(), &event, "OpenLineage run event")?;
        }
        Command::RunMockReplay {
            graph,
            campaign,
            controllers,
            bundle,
            replay_request,
            prediction_cache_payload,
            prediction_cache_store,
            artifact_payload_store,
            envelopes,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
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
            let artifact_store =
                optional_mock_artifact_store(&plan, &bundle, artifact_payload_store.as_ref())?;
            let runtime_controllers = mock_runtime_controllers(&plan)?;
            let mut ctx = RunContext::new(RunId::new(run_id)?, Some(root_seed));
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let results = execute_bundle_replay_with_scheduler(
                scheduler,
                BundleReplayExecution {
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
                "mock replay run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s), {} artifact handle(s), {} prediction cache handle(s), scheduler={}, scheduler worker(s)={}",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len(),
                artifact_store.artifact_handle_count(),
                prediction_cache_store
                    .as_ref()
                    .map(CliPredictionCacheStore::materialization_record_count)
                    .unwrap_or(0),
                scheduler.scheduler.label(),
                scheduler.workers
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
            process_workers,
            process_timeout_ms,
            process_retries,
            plan_id,
            run_id,
            root_seed,
            scheduler,
            scheduler_workers,
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
            let process_config = process_adapter_runtime_config(
                process_workers,
                process_timeout_ms,
                process_retries,
            )?;
            let scheduler = SchedulerConfig::new(scheduler, scheduler_workers)?;
            let runtime_controllers = process_runtime_controllers_for_mode(
                &plan,
                adapter,
                persistent,
                process_config,
                scheduler,
            )?;
            let mut ctx = RunContext::new(RunId::new(run_id)?, Some(root_seed));
            let results = execute_bundle_replay_with_scheduler(
                scheduler,
                BundleReplayExecution {
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
                "process replay run: {} result(s), {} lineage record(s), {} prediction block(s), {} data handle(s), {} data view(s), {} artifact handle(s), {} prediction cache handle(s), scheduler={}, scheduler worker(s)={}, configured process worker(s)={}, observed process worker(s)={}",
                results.len(),
                ctx.lineage.len(),
                ctx.prediction_store.blocks().len(),
                data_provider.handle_records().len(),
                data_provider.view_records().len(),
                artifact_store.len(),
                prediction_cache_store
                    .as_ref()
                    .map(CliPredictionCacheStore::materialization_record_count)
                    .unwrap_or(0),
                scheduler.scheduler.label(),
                scheduler.workers,
                configured_persistent_process_workers(persistent, process_workers),
                observed_persistent_process_worker_count(persistent, &ctx)
            );
        }
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct RegressionScoreCliOutput {
    pub report: RegressionMetricReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_score: Option<CandidateScore>,
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
        required_metric_level: Some(dag_ml_core::PredictionLevel::Sample),
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
                metadata: BTreeMap::from([(
                    "metric_level".to_string(),
                    serde_json::Value::String("sample".to_string()),
                )]),
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
    scheduler: SchedulerConfig,
}

struct CapturedRefitBundle {
    bundle: ExecutionBundle,
    artifact_store: InMemoryArtifactStore,
    lineage_records: Vec<LineageRecord>,
    prediction_cache_payloads: Vec<BundlePredictionCachePayload>,
    fit_cv_result_count: usize,
    fit_cv_oof_prediction_block_count: usize,
    refit_result_count: usize,
    observed_process_worker_count: usize,
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

    let results = execute_campaign_phase_with_artifact_store_and_scheduler(
        input.scheduler,
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
        lineage_records: ctx.lineage.records().cloned().collect(),
        prediction_cache_payloads: Vec::new(),
        fit_cv_result_count: 0,
        fit_cv_oof_prediction_block_count: 0,
        refit_result_count: results.len(),
        observed_process_worker_count: observed_process_worker_count(&ctx),
    })
}

fn build_bundle_from_cv_then_captured_refit(
    input: CapturedRefitBundleInput<'_>,
) -> Result<CapturedRefitBundle> {
    let selected_variant_id = selected_refit_variant(input.plan, input.variant_id)?;

    let mut artifact_store = InMemoryArtifactStore::new();
    let mut ctx = RunContext::new(RunId::new(input.run_id)?, Some(input.root_seed));
    ctx.variant_id = Some(selected_variant_id.clone());

    let fit_cv_results = execute_campaign_phase_with_scheduler(
        input.scheduler,
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

    let refit_results = execute_campaign_phase_with_artifact_store_and_scheduler(
        input.scheduler,
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
        lineage_records: ctx.lineage.records().cloned().collect(),
        prediction_cache_payloads,
        fit_cv_result_count: fit_cv_results.len(),
        fit_cv_oof_prediction_block_count,
        refit_result_count: refit_results.len(),
        observed_process_worker_count: observed_process_worker_count(&ctx),
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
            prediction_level: PredictionLevel::Sample,
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
            unit_ids: Vec::new(),
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
    config: ProcessAdapterRuntimeConfig,
}

#[derive(Debug, Deserialize)]
struct ProcessAdapterDescription {
    schema_version: u32,
    protocol: String,
    adapter_id: String,
    supported_modes: BTreeSet<String>,
    capabilities: BTreeSet<String>,
}

#[derive(Clone, Copy, Debug)]
struct ProcessAdapterRuntimeConfig {
    process_workers: usize,
    timeout: Duration,
    retries: usize,
    control_frames: bool,
}

struct PersistentProcessRuntimeController {
    id: ControllerId,
    adapter: PathBuf,
    config: ProcessAdapterRuntimeConfig,
    sessions: Vec<Mutex<PersistentProcessSession>>,
}

struct PersistentProcessSession {
    child: Child,
    stdin: ChildStdin,
    stdout_rx: Receiver<PersistentReadEvent>,
    control_frames: bool,
    close_timeout: Duration,
}

enum PersistentReadEvent {
    Line(String),
    Eof,
    Error(String),
}

struct PersistentWorkerFailure {
    restartable: bool,
    message: String,
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ProcessAdapterRequestFrame<'a> {
    Init {
        schema_version: u32,
        controller_id: &'a str,
        worker_index: usize,
        worker_count: usize,
    },
    Task {
        schema_version: u32,
        task: &'a NodeTask,
    },
    Close {
        schema_version: u32,
    },
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ProcessAdapterResponseFrame {
    Ack {
        schema_version: u32,
        status: String,
    },
    Result {
        schema_version: u32,
        result: Box<NodeResult>,
    },
    Error {
        schema_version: u32,
        error: ProcessAdapterErrorFrame,
    },
}

#[derive(Debug, Deserialize)]
struct ProcessAdapterErrorFrame {
    code: String,
    message: String,
    #[serde(default)]
    retryable: bool,
}

impl ProcessAdapterResponseFrame {
    fn kind(&self) -> &'static str {
        match self {
            Self::Ack { .. } => "ack",
            Self::Result { .. } => "result",
            Self::Error { .. } => "error",
        }
    }
}

impl PersistentWorkerFailure {
    fn restartable(message: impl Into<String>) -> Self {
        Self {
            restartable: true,
            message: message.into(),
        }
    }

    fn terminal(message: impl Into<String>) -> Self {
        Self {
            restartable: false,
            message: message.into(),
        }
    }
}

impl PersistentProcessSession {
    fn spawn(
        controller_id: &ControllerId,
        adapter: &Path,
        worker_index: usize,
        worker_count: usize,
        control_frames: bool,
        timeout: Duration,
    ) -> dag_ml_core::Result<Self> {
        let mut command = process_adapter_command(adapter, ProcessAdapterMode::Jsonl);
        command.stdin(Stdio::piped());
        command.stdout(Stdio::piped());
        command.stderr(Stdio::inherit());
        command.env("DAG_ML_CONTROLLER_ID", controller_id.as_str());
        command.env("DAG_ML_PROCESS_WORKER_INDEX", worker_index.to_string());
        command.env("DAG_ML_PROCESS_WORKER_COUNT", worker_count.to_string());
        let mut child = command.spawn().map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` failed to spawn persistent adapter `{}` worker {worker_index}/{worker_count}: {err}",
                adapter.display()
            ))
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` persistent adapter worker {worker_index}/{worker_count} stdin was not available"
            ))
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` persistent adapter worker {worker_index}/{worker_count} stdout was not available"
            ))
        })?;
        let mut session = Self {
            child,
            stdin,
            stdout_rx: spawn_persistent_stdout_reader(stdout),
            control_frames,
            close_timeout: timeout.min(Duration::from_millis(250)),
        };
        if control_frames {
            session
                .init(controller_id, adapter, worker_index, worker_count, timeout)
                .map_err(|failure| {
                    session.terminate();
                    DagMlError::RuntimeValidation(format!(
                        "controller `{controller_id}` failed to initialize persistent adapter `{}` worker {worker_index}/{worker_count}: {}",
                        adapter.display(),
                        failure.message
                    ))
                })?;
        }
        Ok(session)
    }

    fn init(
        &mut self,
        controller_id: &ControllerId,
        adapter: &Path,
        worker_index: usize,
        worker_count: usize,
        timeout: Duration,
    ) -> Result<(), PersistentWorkerFailure> {
        self.write_json_line(
            controller_id,
            ProcessAdapterRequestFrame::Init {
                schema_version: PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
                controller_id: controller_id.as_str(),
                worker_index,
                worker_count,
            },
        )?;
        match self.read_response_frame(controller_id, adapter, timeout)? {
            ProcessAdapterResponseFrame::Ack {
                schema_version,
                status,
            } if schema_version == PROCESS_ADAPTER_FRAME_SCHEMA_VERSION
                && status == "initialized" =>
            {
                Ok(())
            }
            ProcessAdapterResponseFrame::Error {
                schema_version,
                error,
            } if schema_version == PROCESS_ADAPTER_FRAME_SCHEMA_VERSION => {
                Err(PersistentWorkerFailure {
                    restartable: error.retryable,
                    message: format!(
                        "adapter init returned error `{}`: {}",
                        error.code, error.message
                    ),
                })
            }
            frame => Err(PersistentWorkerFailure::terminal(format!(
                "adapter init returned unexpected frame `{}`",
                frame.kind()
            ))),
        }
    }

    fn invoke_once(
        &mut self,
        controller_id: &ControllerId,
        adapter: &Path,
        task: &NodeTask,
        timeout: Duration,
    ) -> Result<NodeResult, PersistentWorkerFailure> {
        if self.control_frames {
            return self.invoke_framed(controller_id, adapter, task, timeout);
        }

        self.write_json_line(controller_id, task)?;

        let line = self.read_response_line(controller_id, adapter, timeout)?;
        serde_json::from_str(&line).map_err(|err| {
            PersistentWorkerFailure::terminal(format!(
                "controller `{controller_id}` persistent adapter `{}` returned invalid NodeResult JSON: {err}",
                adapter.display()
            ))
        })
    }

    fn invoke_framed(
        &mut self,
        controller_id: &ControllerId,
        adapter: &Path,
        task: &NodeTask,
        timeout: Duration,
    ) -> Result<NodeResult, PersistentWorkerFailure> {
        self.write_json_line(
            controller_id,
            ProcessAdapterRequestFrame::Task {
                schema_version: PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
                task,
            },
        )?;
        match self.read_response_frame(controller_id, adapter, timeout)? {
            ProcessAdapterResponseFrame::Result {
                schema_version,
                result,
            } if schema_version == PROCESS_ADAPTER_FRAME_SCHEMA_VERSION => Ok(*result),
            ProcessAdapterResponseFrame::Error {
                schema_version,
                error,
            } if schema_version == PROCESS_ADAPTER_FRAME_SCHEMA_VERSION => {
                Err(PersistentWorkerFailure {
                    restartable: error.retryable,
                    message: format!(
                        "adapter task returned error `{}`: {}",
                        error.code, error.message
                    ),
                })
            }
            frame => Err(PersistentWorkerFailure::terminal(format!(
                "adapter task returned unexpected frame `{}`",
                frame.kind()
            ))),
        }
    }

    fn write_json_line<T: Serialize>(
        &mut self,
        controller_id: &ControllerId,
        value: T,
    ) -> Result<(), PersistentWorkerFailure> {
        serde_json::to_writer(&mut self.stdin, &value).map_err(|err| {
            PersistentWorkerFailure::terminal(format!(
                "controller `{controller_id}` failed to serialize persistent adapter JSON: {err}"
            ))
        })?;
        self.stdin.write_all(b"\n").map_err(|err| {
            PersistentWorkerFailure::restartable(format!(
                "controller `{controller_id}` failed to write persistent adapter JSON: {err}"
            ))
        })?;
        self.stdin.flush().map_err(|err| {
            PersistentWorkerFailure::restartable(format!(
                "controller `{controller_id}` failed to flush persistent adapter JSON: {err}"
            ))
        })
    }

    fn read_response_frame(
        &mut self,
        controller_id: &ControllerId,
        adapter: &Path,
        timeout: Duration,
    ) -> Result<ProcessAdapterResponseFrame, PersistentWorkerFailure> {
        let line = self.read_response_line(controller_id, adapter, timeout)?;
        serde_json::from_str(&line).map_err(|err| {
            PersistentWorkerFailure::terminal(format!(
                "controller `{controller_id}` persistent adapter `{}` returned invalid response frame JSON: {err}",
                adapter.display()
            ))
        })
    }

    fn read_response_line(
        &mut self,
        controller_id: &ControllerId,
        adapter: &Path,
        timeout: Duration,
    ) -> Result<String, PersistentWorkerFailure> {
        match self.stdout_rx.recv_timeout(timeout) {
            Ok(PersistentReadEvent::Line(line)) => Ok(line),
            Ok(PersistentReadEvent::Eof) => {
                let status = self
                    .child
                    .try_wait()
                    .map(|status| status.map(|status| status.to_string()))
                    .unwrap_or_else(|err| Some(format!("status unavailable: {err}")))
                    .unwrap_or_else(|| "still running".to_string());
                Err(PersistentWorkerFailure::restartable(format!(
                    "controller `{controller_id}` persistent adapter `{}` returned EOF ({status})",
                    adapter.display()
                )))
            }
            Ok(PersistentReadEvent::Error(err)) => {
                Err(PersistentWorkerFailure::restartable(format!(
                    "controller `{controller_id}` failed to read persistent adapter `{}`: {err}",
                    adapter.display()
                )))
            }
            Err(RecvTimeoutError::Timeout) => {
                self.terminate();
                Err(PersistentWorkerFailure::restartable(format!(
                    "controller `{controller_id}` persistent adapter `{}` timed out after {} ms",
                    adapter.display(),
                    timeout.as_millis()
                )))
            }
            Err(RecvTimeoutError::Disconnected) => {
                Err(PersistentWorkerFailure::restartable(format!(
                    "controller `{controller_id}` persistent adapter `{}` stdout reader disconnected",
                    adapter.display()
                )))
            }
        }
    }

    fn terminate(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn close_gracefully(&mut self) {
        if !self.control_frames {
            return;
        }
        let Ok(controller_id) = ControllerId::new("controller:process.close") else {
            return;
        };
        if self
            .write_json_line(
                &controller_id,
                ProcessAdapterRequestFrame::Close {
                    schema_version: PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
                },
            )
            .is_err()
        {
            return;
        }
        let _ = self.stdout_rx.recv_timeout(self.close_timeout);
    }
}

impl Drop for PersistentProcessSession {
    fn drop(&mut self) {
        self.close_gracefully();
        self.terminate();
    }
}

fn spawn_persistent_stdout_reader(stdout: ChildStdout) -> Receiver<PersistentReadEvent> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stdout);
        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => {
                    let _ = tx.send(PersistentReadEvent::Eof);
                    break;
                }
                Ok(_) => {
                    if tx.send(PersistentReadEvent::Line(line)).is_err() {
                        break;
                    }
                }
                Err(err) => {
                    let _ = tx.send(PersistentReadEvent::Error(err.to_string()));
                    break;
                }
            }
        }
    });
    rx
}

fn spawn_pipe_reader<R>(mut reader: R) -> std::thread::JoinHandle<std::io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    std::thread::spawn(move || {
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        Ok(buffer)
    })
}

fn join_pipe_reader(
    handle: std::thread::JoinHandle<std::io::Result<Vec<u8>>>,
    controller_id: &ControllerId,
    adapter: &Path,
    stream_name: &str,
) -> dag_ml_core::Result<Vec<u8>> {
    handle
        .join()
        .map_err(|_| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` adapter `{}` {stream_name} reader panicked",
                adapter.display()
            ))
        })?
        .map_err(|err| {
            DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` failed to read adapter `{}` {stream_name}: {err}",
                adapter.display()
            ))
        })
}

fn wait_with_output_timeout(
    mut child: Child,
    timeout: Duration,
    controller_id: &ControllerId,
    adapter: &Path,
) -> dag_ml_core::Result<std::process::Output> {
    let stdout = child.stdout.take().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "controller `{controller_id}` adapter `{}` stdout was not available",
            adapter.display()
        ))
    })?;
    let stderr = child.stderr.take().ok_or_else(|| {
        DagMlError::RuntimeValidation(format!(
            "controller `{controller_id}` adapter `{}` stderr was not available",
            adapter.display()
        ))
    })?;
    let stdout_reader = spawn_pipe_reader(stdout);
    let stderr_reader = spawn_pipe_reader(stderr);
    let started_at = Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = join_pipe_reader(stdout_reader, controller_id, adapter, "stdout")?;
                let stderr = join_pipe_reader(stderr_reader, controller_id, adapter, "stderr")?;
                return Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {}
            Err(err) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = join_pipe_reader(stdout_reader, controller_id, adapter, "stdout");
                let _ = join_pipe_reader(stderr_reader, controller_id, adapter, "stderr");
                return Err(DagMlError::RuntimeValidation(format!(
                    "controller `{controller_id}` failed while waiting for adapter `{}`: {err}",
                    adapter.display()
                )));
            }
        }

        let elapsed = started_at.elapsed();
        if elapsed >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = join_pipe_reader(stdout_reader, controller_id, adapter, "stdout");
            let _ = join_pipe_reader(stderr_reader, controller_id, adapter, "stderr");
            return Err(DagMlError::RuntimeValidation(format!(
                "controller `{controller_id}` adapter `{}` timed out after {} ms",
                adapter.display(),
                timeout.as_millis()
            )));
        }
        std::thread::sleep((timeout - elapsed).min(Duration::from_millis(10)));
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

        let output = wait_with_output_timeout(child, self.config.timeout, &self.id, &self.adapter)?;
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
        if self.sessions.is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "controller `{}` persistent adapter pool has no workers",
                self.id
            )));
        }
        let worker_index = process_worker_index_for_task(task, self.sessions.len());
        let mut session = self.sessions[worker_index].lock().map_err(|_| {
            DagMlError::RuntimeValidation(format!(
                "controller `{}` persistent adapter worker {worker_index}/{} is poisoned",
                self.id, self.config.process_workers
            ))
        })?;
        for attempt in 0..=self.config.retries {
            match session.invoke_once(&self.id, &self.adapter, task, self.config.timeout) {
                Ok(result) => return Ok(result),
                Err(failure) => {
                    if failure.restartable {
                        session.terminate();
                        if attempt < self.config.retries {
                            let replacement = PersistentProcessSession::spawn(
                                &self.id,
                                &self.adapter,
                                worker_index,
                                self.config.process_workers,
                                self.config.control_frames,
                                self.config.timeout,
                            )
                            .map_err(|restart_err| {
                                DagMlError::RuntimeValidation(format!(
                                    "{}; additionally failed to restart persistent worker {}/{}: {restart_err}",
                                    failure.message, worker_index, self.config.process_workers
                                ))
                            })?;
                            *session = replacement;
                            continue;
                        }
                    }
                    let attempts = attempt + 1;
                    return Err(DagMlError::RuntimeValidation(format!(
                        "{} after {} attempt(s)",
                        failure.message, attempts
                    )));
                }
            }
        }
        Err(DagMlError::RuntimeValidation(format!(
            "controller `{}` persistent adapter `{}` exhausted retry budget",
            self.id,
            self.adapter.display()
        )))
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
                let artifact_input = task.artifact_inputs.get(key).ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "node `{}` did not receive replay artifact metadata `{key}`",
                        task.node_plan.node_id
                    ))
                })?;
                if artifact_input.node_id != task.node_plan.node_id
                    || artifact_input.controller_id != task.node_plan.controller_id
                {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received mismatched replay artifact metadata `{key}`",
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
                backend: None,
                uri: None,
                content_fingerprint: None,
                size_bytes: Some(128),
                plugin: None,
                plugin_version: None,
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
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
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
    config: ProcessAdapterRuntimeConfig,
    scheduler: SchedulerConfig,
) -> Result<RuntimeControllerRegistry> {
    let description = validate_process_adapter_description(&adapter, ProcessAdapterMode::OneShot)?;
    validate_process_adapter_execution_capabilities(&adapter, &description, false, scheduler)?;
    let mut registry = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        registry.register(Box::new(ProcessRuntimeController {
            id: controller_id.clone(),
            adapter: adapter.clone(),
            config,
        }))?;
    }
    Ok(registry)
}

fn process_runtime_controllers_for_mode(
    plan: &dag_ml_core::ExecutionPlan,
    adapter: PathBuf,
    persistent: bool,
    config: ProcessAdapterRuntimeConfig,
    scheduler: SchedulerConfig,
) -> Result<RuntimeControllerRegistry> {
    validate_process_runtime_config(persistent, &config)?;
    if persistent {
        persistent_process_runtime_controllers(plan, adapter, config, scheduler)
    } else {
        process_runtime_controllers(plan, adapter, config, scheduler)
    }
}

fn persistent_process_runtime_controllers(
    plan: &dag_ml_core::ExecutionPlan,
    adapter: PathBuf,
    mut config: ProcessAdapterRuntimeConfig,
    scheduler: SchedulerConfig,
) -> Result<RuntimeControllerRegistry> {
    validate_process_runtime_config(true, &config)?;
    let description = validate_process_adapter_description(&adapter, ProcessAdapterMode::Jsonl)?;
    validate_process_adapter_execution_capabilities(&adapter, &description, true, scheduler)?;
    config.control_frames = true;
    let mut registry = RuntimeControllerRegistry::new();
    for controller_id in plan.controller_manifests.keys() {
        let mut sessions = Vec::with_capacity(config.process_workers);
        for worker_index in 0..config.process_workers {
            sessions.push(Mutex::new(PersistentProcessSession::spawn(
                controller_id,
                &adapter,
                worker_index,
                config.process_workers,
                config.control_frames,
                config.timeout,
            )?));
        }
        registry.register(Box::new(PersistentProcessRuntimeController {
            id: controller_id.clone(),
            adapter: adapter.clone(),
            config,
            sessions,
        }))?;
    }
    Ok(registry)
}

fn validate_process_adapter_execution_capabilities(
    adapter: &Path,
    description: &ProcessAdapterDescription,
    persistent: bool,
    scheduler: SchedulerConfig,
) -> Result<()> {
    if scheduler.scheduler == CliScheduler::Parallel
        && scheduler.workers > 1
        && !description
            .capabilities
            .contains(PROCESS_ADAPTER_CAP_PARALLEL_INVOCATION)
    {
        bail!(
            "adapter `{}` is missing required parallel scheduler capability `{}`",
            adapter.display(),
            PROCESS_ADAPTER_CAP_PARALLEL_INVOCATION
        );
    }
    if persistent {
        for capability in [
            PROCESS_ADAPTER_CAP_CONTROL_FRAMES,
            PROCESS_ADAPTER_CAP_PERSISTENT_WORKERS,
            PROCESS_ADAPTER_CAP_WORKER_ENV,
        ] {
            if !description.capabilities.contains(capability) {
                bail!(
                    "adapter `{}` is missing required persistent capability `{}`",
                    adapter.display(),
                    capability
                );
            }
        }
    }
    Ok(())
}

fn process_adapter_runtime_config(
    process_workers: usize,
    process_timeout_ms: u64,
    process_retries: usize,
) -> Result<ProcessAdapterRuntimeConfig> {
    if process_timeout_ms == 0 {
        bail!("--process-timeout-ms must be at least 1");
    }
    Ok(ProcessAdapterRuntimeConfig {
        process_workers,
        timeout: Duration::from_millis(process_timeout_ms),
        retries: process_retries,
        control_frames: false,
    })
}

fn validate_process_runtime_config(
    persistent: bool,
    config: &ProcessAdapterRuntimeConfig,
) -> Result<()> {
    if config.process_workers == 0 {
        bail!("--process-workers must be at least 1");
    }
    if !persistent && config.process_workers != 1 {
        bail!("--process-workers > 1 requires --persistent");
    }
    if !persistent && config.retries != 0 {
        bail!("--process-retries requires --persistent");
    }
    Ok(())
}

fn validate_process_adapter_description(
    adapter: &Path,
    mode: ProcessAdapterMode,
) -> Result<ProcessAdapterDescription> {
    let output = process_adapter_command(adapter, ProcessAdapterMode::Describe)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .with_context(|| {
            format!(
                "failed to run adapter `{}` describe handshake",
                adapter.display()
            )
        })?;
    if !output.status.success() {
        bail!(
            "adapter `{}` describe handshake exited with status {}: {}",
            adapter.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    if output.stdout.is_empty() {
        bail!(
            "adapter `{}` describe handshake returned empty stdout",
            adapter.display()
        );
    }
    let description: ProcessAdapterDescription = serde_json::from_slice(&output.stdout)
        .with_context(|| {
            format!(
                "adapter `{}` describe handshake returned invalid JSON",
                adapter.display()
            )
        })?;
    if description.schema_version != PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION {
        bail!(
            "adapter `{}` has unsupported description schema version {}",
            adapter.display(),
            description.schema_version
        );
    }
    if description.protocol != PROCESS_ADAPTER_PROTOCOL {
        bail!(
            "adapter `{}` has unsupported protocol `{}`",
            adapter.display(),
            description.protocol
        );
    }
    if description.adapter_id.trim().is_empty() {
        bail!(
            "adapter `{}` returned an empty adapter_id",
            adapter.display()
        );
    }
    let mode_name = mode.describe_name();
    if !description.supported_modes.contains(mode_name) {
        bail!(
            "adapter `{}` does not support required mode `{}`",
            adapter.display(),
            mode_name
        );
    }
    for capability in [
        PROCESS_ADAPTER_CAP_NODE_TASK_JSON,
        PROCESS_ADAPTER_CAP_NODE_RESULT_JSON,
    ] {
        if !description.capabilities.contains(capability) {
            bail!(
                "adapter `{}` is missing required capability `{}`",
                adapter.display(),
                capability
            );
        }
    }
    Ok(description)
}

fn process_worker_index_for_task(task: &NodeTask, worker_count: usize) -> usize {
    debug_assert!(worker_count > 0);
    let variant = task
        .variant_id
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_else(|| "base".to_string());
    let key = if task.phase == Phase::FitCv {
        format!(
            "{}:{}:{}",
            task.node_plan.node_id,
            variant,
            task.fold_id
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "nofold".to_string())
        )
    } else {
        format!("{}:{}", task.node_plan.node_id, variant)
    };
    (stable_handle(&key) as usize) % worker_count
}

fn observed_process_worker_count(ctx: &RunContext) -> usize {
    ctx.lineage
        .records()
        .filter_map(|record| record.metrics.get("process_worker_index"))
        .map(|value| *value as u64)
        .collect::<BTreeSet<_>>()
        .len()
}

fn configured_persistent_process_workers(persistent: bool, process_workers: usize) -> usize {
    if persistent {
        process_workers
    } else {
        0
    }
}

fn observed_persistent_process_worker_count(persistent: bool, ctx: &RunContext) -> usize {
    if persistent {
        observed_process_worker_count(ctx)
    } else {
        0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProcessAdapterMode {
    OneShot,
    Jsonl,
    Describe,
}

impl ProcessAdapterMode {
    fn describe_name(self) -> &'static str {
        match self {
            Self::OneShot => PROCESS_ADAPTER_MODE_ONE_SHOT,
            Self::Jsonl => PROCESS_ADAPTER_MODE_JSONL,
            Self::Describe => "describe",
        }
    }
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
    match mode {
        ProcessAdapterMode::OneShot => {}
        ProcessAdapterMode::Jsonl => {
            command.arg("--jsonl");
        }
        ProcessAdapterMode::Describe => {
            command.arg("--describe");
        }
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

fn build_plan_from_dsl_path(
    dsl: &PathBuf,
    controllers: &PathBuf,
    plan_id: String,
) -> Result<dag_ml_core::ExecutionPlan> {
    let spec: PipelineDslSpec = read_json(dsl, "pipeline DSL")?;
    let compiled = compile_pipeline_dsl_with_generation(&spec)
        .with_context(|| format!("failed to compile pipeline DSL at {}", dsl.display()))?;
    let registry = controller_registry_from_path(controllers)?;
    build_execution_plan(
        plan_id,
        compiled.graph,
        compiled.campaign_template,
        &registry,
    )
    .with_context(|| {
        format!(
            "failed to build execution plan from pipeline DSL at {}",
            dsl.display()
        )
    })
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

enum CliArtifactStore {
    InMemory(InMemoryArtifactStore),
    File(FileArtifactPayloadStore),
}

impl CliArtifactStore {
    fn artifact_handle_count(&self) -> usize {
        match self {
            Self::InMemory(store) => store.len(),
            Self::File(store) => store.materialization_records().len(),
        }
    }
}

impl RuntimeArtifactStore for CliArtifactStore {
    fn materialize(
        &self,
        request: &dag_ml_core::ArtifactMaterializationRequest,
    ) -> dag_ml_core::Result<HandleRef> {
        match self {
            Self::InMemory(store) => store.materialize(request),
            Self::File(store) => store.materialize(request),
        }
    }
}

fn optional_mock_artifact_store(
    plan: &dag_ml_core::ExecutionPlan,
    bundle: &ExecutionBundle,
    payload_store_dir: Option<&PathBuf>,
) -> Result<CliArtifactStore> {
    if let Some(store_dir) = payload_store_dir {
        return validate_file_artifact_payload_store(bundle, store_dir).map(CliArtifactStore::File);
    }
    mock_artifact_store(plan, bundle).map(CliArtifactStore::InMemory)
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

    fn load_aggregated_blocks(
        &self,
        requirement_key: &str,
    ) -> dag_ml_core::Result<Vec<dag_ml_core::AggregatedPredictionBlock>> {
        match self {
            Self::Columnar(store) => store.load_aggregated_blocks(requirement_key),
            Self::File(store) => store.load_aggregated_blocks(requirement_key),
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
        if entry.prediction_level == PredictionLevel::Sample {
            store.load_blocks(&entry.requirement_key).with_context(|| {
                format!(
                    "prediction cache store cannot load `{}`",
                    entry.requirement_key
                )
            })?;
        } else {
            store
                .load_aggregated_blocks(&entry.requirement_key)
                .with_context(|| {
                    format!(
                        "prediction cache store cannot load aggregated `{}`",
                        entry.requirement_key
                    )
                })?;
        }
    }
    Ok(store)
}

fn validate_file_artifact_manifest_store(
    bundle: &ExecutionBundle,
    manifest_dir: &Path,
) -> Result<FileArtifactManifestStore> {
    FileArtifactManifestStore::open(manifest_dir.to_path_buf(), bundle).with_context(|| {
        format!(
            "artifact manifest store is invalid at {}",
            manifest_dir.display()
        )
    })
}

fn validate_file_artifact_payload_store(
    bundle: &ExecutionBundle,
    store_dir: &Path,
) -> Result<FileArtifactPayloadStore> {
    let store =
        FileArtifactPayloadStore::open(store_dir.to_path_buf(), bundle).with_context(|| {
            format!(
                "artifact payload store is invalid at {}",
                store_dir.display()
            )
        })?;
    store.validate_payloads().with_context(|| {
        format!(
            "artifact payload store cannot validate payloads at {}",
            store_dir.display()
        )
    })?;
    Ok(store)
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf, label: &str) -> Result<T> {
    let data = std::fs::read(path)
        .with_context(|| format!("failed to read {label} JSON at {}", path.display()))?;
    serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse {label} JSON at {}", path.display()))
}

fn write_research_provenance_package(
    output_dir: &Path,
    package: &ResearchProvenancePackage,
) -> Result<()> {
    std::fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "failed to create research provenance output dir {}",
            output_dir.display()
        )
    })?;
    for file in package.files.values() {
        let relative_path = Path::new(&file.path);
        let safe_relative_path = !relative_path.is_absolute()
            && relative_path
                .components()
                .all(|component| matches!(component, Component::Normal(_)));
        if !safe_relative_path {
            bail!(
                "research provenance package contains unsafe output path `{}`",
                file.path
            );
        }
        let output = output_dir.join(relative_path);
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create research provenance package directory {}",
                    parent.display()
                )
            })?;
        }
        std::fs::write(&output, &file.bytes).with_context(|| {
            format!(
                "failed to write research provenance package file {}",
                output.display()
            )
        })?;
    }
    Ok(())
}

fn read_research_provenance_package_dir(input_dir: &Path) -> Result<BTreeMap<String, Vec<u8>>> {
    if !input_dir.is_dir() {
        bail!(
            "research provenance package dir `{}` is not a directory",
            input_dir.display()
        );
    }
    let mut files = BTreeMap::new();
    read_research_provenance_package_dir_inner(input_dir, input_dir, &mut files)?;
    if files.is_empty() {
        bail!(
            "research provenance package dir `{}` contains no files",
            input_dir.display()
        );
    }
    Ok(files)
}

fn read_research_provenance_package_dir_inner(
    root: &Path,
    current: &Path,
    files: &mut BTreeMap<String, Vec<u8>>,
) -> Result<()> {
    let mut entries = std::fs::read_dir(current)
        .with_context(|| {
            format!(
                "failed to read research provenance package directory {}",
                current.display()
            )
        })?
        .collect::<std::result::Result<Vec<_>, _>>()
        .with_context(|| {
            format!(
                "failed to enumerate research provenance package directory {}",
                current.display()
            )
        })?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let entry_path = entry.path();
        let file_type = entry.file_type().with_context(|| {
            format!(
                "failed to inspect research provenance package path {}",
                entry_path.display()
            )
        })?;
        if file_type.is_dir() {
            read_research_provenance_package_dir_inner(root, &entry_path, files)?;
            continue;
        }
        if !file_type.is_file() {
            bail!(
                "research provenance package path `{}` is not a regular file",
                entry_path.display()
            );
        }
        let relative = entry_path.strip_prefix(root).with_context(|| {
            format!(
                "failed to relativize research provenance package file {}",
                entry_path.display()
            )
        })?;
        let package_path = package_relative_path(relative)?;
        let previous = files.insert(
            package_path.clone(),
            std::fs::read(&entry_path).with_context(|| {
                format!(
                    "failed to read research provenance package file {}",
                    entry_path.display()
                )
            })?,
        );
        if previous.is_some() {
            bail!("duplicate research provenance package file `{package_path}`");
        }
    }
    Ok(())
}

fn package_relative_path(path: &Path) -> Result<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        let Component::Normal(part) = component else {
            bail!(
                "research provenance package contains unsafe relative path `{}`",
                path.display()
            );
        };
        let part = part.to_str().ok_or_else(|| {
            anyhow::anyhow!(
                "research provenance package path `{}` is not valid UTF-8",
                path.display()
            )
        })?;
        parts.push(part.to_string());
    }
    if parts.is_empty() {
        bail!("research provenance package contains empty relative path");
    }
    Ok(parts.join("/"))
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
