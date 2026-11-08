//! Captured host HPO through the actual coordinator, Recorder and `verify_stream`.
//!
//! The proposal source is a scripted owner. These tests qualify coordinator
//! stream, cleanup, pairing and resume decisions; real n4m/Optuna behavior needs
//! its own production-path probes.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dag_ml_core::error::{
    HostHpoCaptureAbortReport, HostHpoCaptureDisposition, HostHpoCaptureFlush,
};
use dag_ml_core::realization_evidence::{
    inspect_stream, verify_stream, ChoiceSemantics, CleanupState, Consumption, Counts, Event,
    FailedAttempt, FailureStage, Limits, OwnedAttempt, Ownership, Policy, Recorder, ResolutionKind,
    ScopeKind, SeedValue, TrialStatus, ValueSlot,
};
use dag_ml_core::*;
use serde_json::{json, Value};

const DECLARATION: &str = r#"{"search":{"n_components":[1,2,3]}}"#;
const DECLARATION_DIGEST: &str = "a46fbd129a3f4265ddb6515a22f39f72d14de5b71faecbbce7b3f938cfe206e2";
const SCOPE: &str = "declaration:search";
const SEARCH: &str = "search:capture";
const SEED: &str = "seed:sampler";
const DATA: &str = "capture-regression-data";
const TRIAL_0: &str = "host_hpo:trial:0000000000";
const TRIAL_1: &str = "host_hpo:trial:0000000001";
const PLANNING_SIBLING_HEX: &str =
    "6e6f74206576616c75617465643b206f776e696e672077696e646f7720706c616e6e696e67206661696c6564";

fn hex(text: &str) -> String {
    text.bytes().map(|byte| format!("{byte:02x}")).collect()
}

fn owner_ref(trial_index: u32) -> String {
    format!("{:02x}", 0xa0 + trial_index)
}

fn policy(id: &str) -> Policy {
    Policy {
        id: id.into(),
        version: "1".into(),
        context_hex: hex("policy-context"),
    }
}

#[derive(Clone)]
enum Step {
    Propose(i64),
    /// The owner already reports this terminal status when the ask returns.
    ProposeTerminal(i64, TrialStatus),
    /// Empty parameters refuse in candidate planning, after the ask returned.
    Unplannable,
    RaiseAsk,
    Fail {
        stage: FailureStage,
        owned: Option<OwnedAttempt>,
    },
    Exhausted,
}

#[derive(Clone, Copy, PartialEq)]
enum SeedMode {
    Generated,
    Explicit(u64),
    Deterministic,
    Derived,
}

struct Source {
    steps: BTreeMap<u32, Step>,
    mode: SeedMode,
    seed_resolved: bool,
    /// Resolve the seed again on every proposal, with a changing value.
    repeat_resolution: bool,
    asks: Vec<u32>,
    tells: Vec<String>,
    pruned: Vec<String>,
    /// Every cleanup request, whether or not it transitioned the owner.
    fails: Vec<(String, Option<FailureStage>, String)>,
    /// Genuine owner FAILED transitions only.
    transitions: Vec<String>,
    prune_on_report: BTreeSet<String>,
    /// The owner store: every trial and its status.
    store: BTreeMap<String, TrialStatus>,
    /// Sampler position, which moves independently of the rows.
    rng_state: u64,
    /// Parameters of asked trials, which the digest covers while they run.
    trial_params: BTreeMap<String, i64>,
    /// The owner's own failure text, when it differs from what it was told.
    owner_diagnostic: Option<String>,
    /// Added to a stated seed value, as an owner that drifted would.
    seed_shift: u64,
    /// State the same seed twice in one proposal.
    duplicate_resolution: bool,
    /// Exhausting the owner advances its sampler.
    exhaustion_moves_state: bool,
    refuse_transition: BTreeSet<String>,
    refuse_tell: bool,
    state_unavailable: bool,
    pinned_fingerprint: Option<String>,
}

impl Source {
    fn scripted(steps: Vec<Step>) -> Self {
        Self {
            steps: steps
                .into_iter()
                .enumerate()
                .map(|(index, step)| (index as u32, step))
                .collect(),
            mode: SeedMode::Generated,
            seed_resolved: false,
            repeat_resolution: false,
            asks: Vec::new(),
            tells: Vec::new(),
            pruned: Vec::new(),
            fails: Vec::new(),
            transitions: Vec::new(),
            prune_on_report: BTreeSet::new(),
            store: BTreeMap::new(),
            rng_state: 0,
            trial_params: BTreeMap::new(),
            owner_diagnostic: None,
            seed_shift: 0,
            duplicate_resolution: false,
            exhaustion_moves_state: false,
            refuse_transition: BTreeSet::new(),
            refuse_tell: false,
            state_unavailable: false,
            pinned_fingerprint: None,
        }
    }

    /// A new process over an existing owner store.
    fn resumed(steps: Vec<Step>, store: BTreeMap<String, TrialStatus>) -> Self {
        Self {
            rng_state: store.len() as u64,
            store,
            ..Self::scripted(steps)
        }
    }

    /// The owner store exactly as one publication observed it.
    fn at_publication(publication: &HostHpoCapturePublication, steps: Vec<Step>) -> Self {
        let checkpoint = &publication.checkpoint;
        let store = checkpoint
            .attempts
            .iter()
            .chain(checkpoint.pending_attempts.iter().map(|p| &p.attempt))
            .map(|attempt| {
                (
                    attempt.owner_ref_hex.clone(),
                    attempt.owner_observation.status.unwrap(),
                )
            })
            .collect();
        let mut source = Self::resumed(steps, store);
        for pending in &checkpoint.pending_attempts {
            if let Some(value) = pending.params["n_components"].as_i64() {
                source
                    .trial_params
                    .insert(pending.attempt.owner_ref_hex.clone(), value);
            }
        }
        source
    }

    /// Canonical TCV1 digest of everything that can change a future proposal.
    fn state_digest(&self) -> String {
        let running = self
            .trial_params
            .iter()
            .filter(|(owner, _)| self.store.get(*owner) == Some(&TrialStatus::Running))
            .collect::<BTreeMap<_, _>>();
        let document = json!({
            "statuses": format!("{:?}", self.store),
            "rng_state": self.rng_state,
            "pending_params": running,
        });
        tcv1_sha256(&parse_typed_json(&document.to_string()).unwrap()).unwrap()
    }

    fn seed_facts(
        &mut self,
        choice_id: &str,
    ) -> (Option<String>, ChoiceSemantics, Vec<HostHpoSeedResolution>) {
        let fresh = !self.seed_resolved || self.repeat_resolution;
        let generated = |seed_id: &str, value: u64| HostHpoSeedResolution {
            seed_id: seed_id.into(),
            effective: SeedValue::U64 { value },
            resolution: ResolutionKind::Generated,
            parent: None,
            policy: Some(policy("policy:sampler")),
            first_actual_use_id: choice_id.into(),
        };
        let step = self.asks.len() as u64;
        let (seed_id, resolutions) = match self.mode {
            SeedMode::Deterministic => {
                return (None, ChoiceSemantics::Deterministic, Vec::new());
            }
            SeedMode::Generated if self.duplicate_resolution => {
                (SEED, vec![generated(SEED, 41 + step), generated(SEED, 99)])
            }
            SeedMode::Generated => (SEED, vec![generated(SEED, 41 + step + self.seed_shift)]),
            SeedMode::Explicit(value) => (
                SEED,
                vec![HostHpoSeedResolution {
                    seed_id: SEED.into(),
                    effective: SeedValue::U64 { value },
                    resolution: ResolutionKind::Explicit,
                    parent: None,
                    policy: None,
                    first_actual_use_id: choice_id.into(),
                }],
            ),
            SeedMode::Derived => (
                "seed:child",
                vec![
                    generated("seed:root", 41),
                    HostHpoSeedResolution {
                        seed_id: "seed:child".into(),
                        effective: SeedValue::U64 { value: 43 },
                        resolution: ResolutionKind::Derived,
                        parent: Some(HostHpoSeedParent {
                            seed_id: "seed:root".into(),
                            value_slot: ValueSlot::Effective,
                        }),
                        policy: Some(policy("policy:derive")),
                        first_actual_use_id: choice_id.into(),
                    },
                ],
            ),
        };
        self.seed_resolved = true;
        (
            Some(seed_id.to_owned()),
            ChoiceSemantics::Stochastic,
            if fresh { resolutions } else { Vec::new() },
        )
    }

    fn proposed(&mut self, trial_index: u32, value: Option<i64>) -> HostHpoProposalOutcome {
        let owner_ref_hex = owner_ref(trial_index);
        self.store
            .entry(owner_ref_hex.clone())
            .or_insert(TrialStatus::Running);
        let choice_id = format!("choice:{trial_index}");
        let (params, located_choices, seed_resolutions) = match value {
            Some(value) => {
                self.trial_params.insert(owner_ref_hex.clone(), value);
                let (seed_id, semantics, resolutions) = self.seed_facts(&choice_id);
                (
                    BTreeMap::from([("n_components".to_string(), json!(value))]),
                    vec![HostHpoLocatedChoice {
                        choice_id,
                        parameter: "n_components".into(),
                        node_id: "model:pls".into(),
                        source_path_hex: vec![hex("n_components")],
                        value_hex: hex(&value.to_string()),
                        semantics,
                        seed_id,
                    }],
                    resolutions,
                )
            }
            None => (BTreeMap::new(), Vec::new(), Vec::new()),
        };
        HostHpoProposalOutcome::Proposed {
            owner_evidence_hex: hex(&format!("owner-trial:{trial_index}")),
            owner_ref_hex,
            params,
            located_choices,
            seed_resolutions,
            seed_uses: Vec::new(),
        }
    }
}

impl HostHpoCaptureProposalSource for Source {
    fn proposal_state(&mut self) -> Result<HostHpoProposalState> {
        let digest = self
            .pinned_fingerprint
            .clone()
            .unwrap_or_else(|| self.state_digest());
        Ok(HostHpoProposalState {
            owner_checkpoint_fingerprint: (!self.state_unavailable).then_some(digest),
            structural_binding: None,
            owner_trial_count: self.store.len() as u64,
        })
    }

    fn ask_evidence(
        &mut self,
        trial_index: u32,
        _phase_index: Option<u32>,
    ) -> Result<HostHpoProposalEvidence> {
        self.asks.push(trial_index);
        let step = self
            .steps
            .get(&trial_index)
            .cloned()
            .unwrap_or(Step::Exhausted);
        if !matches!(step, Step::Exhausted) || self.exhaustion_moves_state {
            self.rng_state += 1;
        }
        let outcome = match step {
            Step::Propose(value) => self.proposed(trial_index, Some(value)),
            Step::ProposeTerminal(value, status) => {
                self.store.insert(owner_ref(trial_index), status);
                self.proposed(trial_index, Some(value))
            }
            Step::Unplannable => self.proposed(trial_index, None),
            Step::RaiseAsk => {
                return Err(DagMlError::RuntimeValidation(
                    "owner ask raised after possible side effects".into(),
                ))
            }
            Step::Fail { stage, owned } => {
                if let Some(owner) = owned.as_ref().and_then(|a| a.owner_ref_hex.clone()) {
                    self.store.insert(owner, TrialStatus::Running);
                }
                HostHpoProposalOutcome::Failed {
                    failure_stage: stage,
                    causal_diagnostic_hex: hex("bridge stage failed"),
                    consumption: Consumption::Observed,
                    owned_attempt: owned,
                }
            }
            Step::Exhausted => HostHpoProposalOutcome::Exhausted {},
        };
        Ok(HostHpoProposalEvidence {
            schema_version: 1,
            outcome,
        })
    }

    fn tell(&mut self, owner_ref_hex: &str, _score: f64) -> Result<()> {
        if self.refuse_tell {
            return Err(DagMlError::RuntimeValidation("owner tell refused".into()));
        }
        self.tells.push(owner_ref_hex.to_owned());
        self.store
            .insert(owner_ref_hex.to_owned(), TrialStatus::Completed);
        Ok(())
    }

    fn report_intermediate(&mut self, owner_ref_hex: &str, _: u32, _: f64) -> Result<bool> {
        Ok(self.prune_on_report.contains(owner_ref_hex))
    }

    fn pruned(&mut self, owner_ref_hex: &str) -> Result<()> {
        self.pruned.push(owner_ref_hex.to_owned());
        self.store
            .insert(owner_ref_hex.to_owned(), TrialStatus::Pruned);
        Ok(())
    }

    fn fail(
        &mut self,
        owner_ref_hex: &str,
        stage: Option<FailureStage>,
        diagnostic: &str,
    ) -> Result<OwnedAttempt> {
        self.fails
            .push((owner_ref_hex.to_owned(), stage, diagnostic.to_owned()));
        let status = self.store.get(owner_ref_hex).copied();
        if status != Some(TrialStatus::Running) {
            return Ok(OwnedAttempt {
                owner_ref_hex: Some(owner_ref_hex.to_owned()),
                ownership: Ownership::Confirmed,
                status,
                cleanup: CleanupState::NotAttempted,
                diagnostic_hex: None,
            });
        }
        if self.refuse_transition.contains(owner_ref_hex) {
            return Err(DagMlError::RuntimeValidation(
                "owner refused Failed transition".into(),
            ));
        }
        self.transitions.push(owner_ref_hex.to_owned());
        self.store
            .insert(owner_ref_hex.to_owned(), TrialStatus::Failed);
        Ok(OwnedAttempt {
            owner_ref_hex: Some(owner_ref_hex.to_owned()),
            ownership: Ownership::Confirmed,
            status: Some(TrialStatus::Failed),
            cleanup: CleanupState::Confirmed,
            diagnostic_hex: self.owner_diagnostic.clone(),
        })
    }
}

#[derive(Default)]
struct Progress {
    prepared: Vec<HostHpoCaptureCheckpoint>,
    published: Vec<(HostHpoCapturePublication, HostHpoSearchStatus)>,
    /// Request cancellation once this many attempts are published.
    cancel_after: Option<usize>,
}

impl Progress {
    /// Publications that carry at least one attempt.
    fn seals(&self) -> usize {
        self.published
            .iter()
            .filter(|(publication, _)| !publication.checkpoint.attempts.is_empty())
            .count()
    }

    fn last(&self) -> HostHpoCapturePublication {
        self.published.last().unwrap().0.clone()
    }
}

impl HostHpoCaptureProgress for Progress {
    fn prepare_terminal(
        &mut self,
        checkpoint: &HostHpoCaptureCheckpoint,
        _status: HostHpoSearchStatus,
    ) -> Result<()> {
        checkpoint.verify_seal()?;
        self.prepared.push(checkpoint.clone());
        Ok(())
    }

    fn checkpoint(
        &mut self,
        publication: &HostHpoCapturePublication,
        status: HostHpoSearchStatus,
    ) -> Result<bool> {
        publication.checkpoint.verify_seal()?;
        self.published.push((publication.clone(), status));
        Ok(self.cancel_after != Some(publication.checkpoint.attempts.len()))
    }
}

/// Fails every write, as a full disk or a closed pipe would.
struct BrokenSink;

impl Write for BrokenSink {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::Error::other("sink refused"))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Err(std::io::Error::other("sink refused"))
    }
}

struct Controller {
    id: ControllerId,
    model: bool,
    failing_variant: Option<&'static str>,
    invocations: Arc<AtomicUsize>,
}

impl RuntimeController for Controller {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        let variant_label = task
            .variant_id
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| "base".to_string());
        if self.model && self.failing_variant == Some(variant_label.as_str()) {
            return Err(DagMlError::RuntimeValidation(
                "deliberate evaluation failure".into(),
            ));
        }
        let fold_label = task
            .fold_id
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| "nofold".to_string());
        let sample = match fold_label.as_str() {
            "fold:0" => Some(("s1", 1.0)),
            "fold:1" => Some(("s2", 2.0)),
            _ => None,
        };
        let mut predictions = Vec::new();
        let mut regression_targets = Vec::new();
        if let (true, Some((sample, y_true))) = (self.model, sample) {
            let sample_id = SampleId::new(sample)?;
            let offset = task
                .node_plan
                .params
                .get("n_components")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            predictions.push(PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                producer_port: None,
                partition: PredictionPartition::Validation,
                fold_id: task.fold_id.clone(),
                sample_ids: vec![sample_id.clone()],
                values: vec![vec![y_true + offset]],
                target_names: vec!["y".to_string()],
            });
            regression_targets.push(RegressionTargetBlock {
                validity_masks: None,
                level: PredictionLevel::Sample,
                unit_ids: vec![PredictionUnitId::Sample(sample_id)],
                values: vec![vec![y_true]],
                target_names: vec!["y".to_string()],
            });
        }
        let handle = |kind| HandleRef {
            handle: if self.model { 2 } else { 1 },
            kind,
            owner_controller: self.id.clone(),
        };
        Ok(NodeResult {
            schema_version: None,
            classification_probabilities: Vec::new(),
            consumed_data_views: BTreeMap::new(),
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([
                ("x".to_string(), handle(HandleKind::Data)),
                ("pred".to_string(), handle(HandleKind::Prediction)),
            ]),
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets,
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:{}:{:?}:{variant_label}:{fold_label}",
                    task.node_plan.node_id, task.phase
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
                loss_attestations: Vec::new(),
                early_stopping_records: Vec::new(),
                target_domain: None,
            },
        })
    }
}

fn controllers(
    failing_variant: Option<&'static str>,
    invocations: &Arc<AtomicUsize>,
) -> Result<RuntimeControllerRegistry> {
    let mut registry = RuntimeControllerRegistry::new();
    for (id, model) in [("controller:transform", false), ("controller:model", true)] {
        registry.register(Box::new(Controller {
            id: ControllerId::new(id)?,
            model,
            failing_variant,
            invocations: invocations.clone(),
        }))?;
    }
    Ok(registry)
}

struct ProviderFactory {
    fail: bool,
}

impl HostHpoCandidateProviderFactory for ProviderFactory {
    fn create(&self, _trial_index: u32) -> Result<Box<dyn RuntimeDataProvider + Send>> {
        if self.fail {
            return Err(DagMlError::RuntimeValidation("provider refused".into()));
        }
        Ok(Box::new(InMemoryDataProvider::new(ControllerId::new(
            "controller:data",
        )?)))
    }
}

struct ControllerFactory {
    fail: bool,
    failing_variant: Option<&'static str>,
    created: AtomicUsize,
    invocations: Arc<AtomicUsize>,
}

impl ControllerFactory {
    fn new(failing_variant: Option<&'static str>) -> Self {
        Self {
            fail: false,
            failing_variant,
            created: AtomicUsize::new(0),
            invocations: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl HostHpoCandidateControllerFactory for ControllerFactory {
    fn create(&self, _trial_index: u32) -> Result<RuntimeControllerRegistry> {
        if self.fail {
            return Err(DagMlError::RuntimeValidation("controller refused".into()));
        }
        self.created.fetch_add(1, Ordering::SeqCst);
        controllers(self.failing_variant, &self.invocations)
    }
}

fn port(name: &str, kind: PortKind) -> PortSpec {
    PortSpec {
        name: name.to_string(),
        kind,
        representation: None,
        cardinality: PortCardinality::One,
        unit_level: None,
        alignment_key: None,
        target_level: None,
        description: String::new(),
    }
}

fn node(id: &str, kind: NodeKind, inputs: Vec<PortSpec>, outputs: Vec<PortSpec>) -> NodeSpec {
    NodeSpec {
        id: NodeId::new(id).unwrap(),
        kind,
        operator: None,
        params: BTreeMap::new(),
        ports: PortSchema { inputs, outputs },
        metadata: BTreeMap::new(),
        seed_label: None,
    }
}

fn manifest(id: &str, kind: NodeKind) -> ControllerManifest {
    let mut capabilities = BTreeSet::from([
        ControllerCapability::Deterministic,
        ControllerCapability::ThreadSafe,
        ControllerCapability::ProcessSafe,
    ]);
    if kind == NodeKind::Model {
        capabilities.insert(ControllerCapability::EmitsPredictions);
        capabilities.insert(ControllerCapability::ConsumesOofPredictions);
        capabilities.insert(ControllerCapability::EmitsArtifacts);
        capabilities.insert(ControllerCapability::Stateful);
    }
    ControllerManifest {
        controller_id: ControllerId::new(id).unwrap(),
        controller_version: "0.1.0".to_string(),
        operator_kind: kind,
        priority: 0,
        supported_phases: BTreeSet::from([Phase::FitCv]),
        input_ports: Vec::new(),
        output_ports: Vec::new(),
        data_requirements: None,
        capabilities,
        operator_selectors: Vec::new(),
        fit_scope: ControllerFitScope::FoldTrain,
        rng_policy: RngPolicy::UsesCoreSeed,
        artifact_policy: ArtifactPolicy::Serializable,
    }
}

fn fold(id: &str, train: &str, validation: &str) -> FoldAssignment {
    FoldAssignment {
        fold_id: FoldId::new(id).unwrap(),
        train_sample_ids: vec![SampleId::new(train).unwrap()],
        validation_sample_ids: vec![SampleId::new(validation).unwrap()],
        metadata: BTreeMap::new(),
    }
}

fn plan() -> ExecutionPlan {
    let graph = GraphSpec {
        id: "g".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![
            node(
                "transform:snv",
                NodeKind::Transform,
                vec![],
                vec![port("x", PortKind::Data)],
            ),
            node(
                "model:pls",
                NodeKind::Model,
                vec![port("x", PortKind::Data)],
                vec![port("pred", PortKind::Prediction)],
            ),
        ],
        edges: vec![EdgeSpec {
            source: PortRef {
                node_id: NodeId::new("transform:snv").unwrap(),
                port_name: "x".to_string(),
            },
            target: PortRef {
                node_id: NodeId::new("model:pls").unwrap(),
                port_name: "x".to_string(),
            },
            contract: EdgeContract {
                requires_oof: false,
                requires_fold_alignment: false,
                ..EdgeContract::new(PortKind::Data, None)
            },
        }],
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    };
    let campaign = CampaignSpec {
        inner_cv: None,
        id: "campaign:host_hpo.capture".to_string(),
        root_seed: Some(7),
        leakage_policy: Default::default(),
        aggregation_policy: Default::default(),
        split_invocation: Some(SplitInvocation {
            id: "split:outer".to_string(),
            controller_id: None,
            leakage_policy: Default::default(),
            params: BTreeMap::new(),
            fold_set: Some(FoldSet {
                id: "outer".to_string(),
                sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
                folds: vec![fold("fold:0", "s2", "s1"), fold("fold:1", "s1", "s2")],
                sample_groups: BTreeMap::new(),
                partition_mode: FoldPartitionMode::Partition,
                train_exclusion: FoldTrainExclusion::Relations,
                target_missing_sample_ids: Vec::new(),
            }),
        }),
        generation: GenerationSpec::default(),
        shape_plans: BTreeMap::new(),
        data_bindings: BTreeMap::new(),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    };
    let mut manifests = ControllerRegistry::new();
    manifests
        .register(manifest("controller:transform", NodeKind::Transform))
        .unwrap();
    manifests
        .register(manifest("controller:model", NodeKind::Model))
        .unwrap();
    build_execution_plan("plan:host_hpo:capture", graph, campaign, &manifests).unwrap()
}

fn search_request() -> HostHpoSearchRequest {
    serde_json::from_value(json!({
        "target_node": "model:pls", "trial_budget": 3, "metric": "rmse",
        "direction": "minimize",
        "optimizer_descriptor": {"sampler": "random", "seed": 42}
    }))
    .unwrap()
}

fn declared(seed_id: &str) -> HostHpoSeedDeclaration {
    HostHpoSeedDeclaration {
        seed_id: seed_id.into(),
        requested: None,
    }
}

fn captured(capture_id: &str, failed_attempts: Vec<FailedAttempt>) -> HostHpoCaptureSearchRequest {
    HostHpoCaptureSearchRequest {
        schema_version: 2,
        request: search_request(),
        capture: HostHpoCaptureDeclaration {
            capture_id: capture_id.into(),
            scope_id: SCOPE.into(),
            search_scope_id: SEARCH.into(),
            declaration_text: DECLARATION.into(),
            source_digest: DECLARATION_DIGEST.into(),
            node_ids: vec!["transform:snv".into(), "model:pls".into()],
            seed_declarations: vec![declared(SEED)],
            failed_attempts,
        },
    }
}

fn budget(capture_id: &str, trial_budget: u32) -> HostHpoCaptureSearchRequest {
    let mut request = captured(capture_id, Vec::new());
    request.request.trial_budget = trial_budget;
    request
}

fn options(publication: Option<HostHpoCapturePublication>) -> HostHpoCaptureResumeOptions {
    HostHpoCaptureResumeOptions {
        data_fingerprint: DATA.into(),
        publication,
    }
}

fn report(error: DagMlError) -> HostHpoCaptureAbortReport {
    match error {
        DagMlError::HostHpoCaptureAbort(report) => *report,
        other => panic!("expected a typed capture abort, observed {other}"),
    }
}

fn refusal_code(error: &DagMlError) -> String {
    match error {
        DagMlError::HostHpoPredecessorRefused(detail) => {
            detail.split(':').next().unwrap_or_default().to_owned()
        }
        other => panic!("expected a typed predecessor refusal, observed {other}"),
    }
}

fn run_serial(
    request: &HostHpoCaptureSearchRequest,
    source: &mut Source,
    failing_variant: Option<&'static str>,
    sink: Box<dyn Write + '_>,
    durable: Option<(&HostHpoCaptureResumeOptions, &mut Progress)>,
) -> Result<HostHpoCaptureOutcome> {
    let invocations = Arc::new(AtomicUsize::new(0));
    let registry = controllers(failing_variant, &invocations)?;
    let provider = InMemoryDataProvider::new(ControllerId::new("controller:data")?);
    SequentialScheduler.execute_captured_host_hpo_search(
        &plan(),
        &registry,
        &provider,
        None,
        None,
        request,
        source,
        sink,
        durable.map(|(options, progress)| (options, progress as &mut dyn HostHpoCaptureProgress)),
    )
}

fn run_parallel(
    request: &HostHpoCaptureSearchRequest,
    source: &mut Source,
    factory: &ControllerFactory,
    sink: Box<dyn Write + '_>,
    workers: usize,
    durable: Option<(&HostHpoCaptureResumeOptions, &mut Progress)>,
) -> Result<HostHpoCaptureOutcome> {
    SequentialScheduler.execute_captured_parallel_host_hpo_search(
        &plan(),
        &ProviderFactory { fail: false },
        factory,
        request,
        source,
        sink,
        workers,
        durable.map(|(options, progress)| (options, progress as &mut dyn HostHpoCaptureProgress)),
    )
}

/// The owner-side acceptance: the real verifier with the returned commitment.
fn verified_counts(buffer: &[u8], outcome: &HostHpoCaptureOutcome) -> Counts {
    let completion = outcome.completion.as_ref().expect("a real completion");
    let text = std::str::from_utf8(buffer).unwrap();
    let verified = verify_stream(text, &completion.expected_commitment, Limits::default()).unwrap();
    assert_eq!(verified.completion(), &completion.completion);
    verified.completion().counts.clone()
}

/// The events inside a written stream, in order.
fn stream_events(buffer: &[u8]) -> Vec<Value> {
    let mut payload = Vec::new();
    for line in std::str::from_utf8(buffer).unwrap().lines() {
        let record: Value = serde_json::from_str(line).unwrap();
        if let Some(frame) = record["payload_hex"].as_str() {
            for index in (0..frame.len()).step_by(2) {
                payload.push(u8::from_str_radix(&frame[index..index + 2], 16).unwrap());
            }
        }
    }
    std::str::from_utf8(&payload)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn assert_no_completion(buffer: &[u8]) {
    let text = std::str::from_utf8(buffer).unwrap();
    assert!(
        inspect_stream(text, Limits::default()).is_err(),
        "an aborted stream must never verify"
    );
}

/// Produced provenance must pass the actual closed recorder gate.
fn assert_closed_wire(failed: &FailedAttempt) {
    let mut recorder =
        Recorder::new(std::io::sink(), "capture:wire.check", Limits::default()).unwrap();
    recorder
        .record(&Event::Declaration {
            scope_id: SCOPE.into(),
            scope_kind: ScopeKind::Declaration,
            parent_scope_id: None,
            declaration_text: DECLARATION.into(),
            source_digest: DECLARATION_DIGEST.into(),
            node_ids: vec!["model:pls".into()],
        })
        .unwrap();
    recorder
        .record(&Event::CaptureProvenance {
            scope_id: SCOPE.into(),
            failed_attempts: vec![failed.clone()],
        })
        .unwrap();
}

#[test]
fn two_completed_trials_finish_with_a_verified_completion() {
    let request = budget("capture:first", 2);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut buffer = Vec::new();
    let outcome = run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap();
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Completed);
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(
        counts.scope_count, 6,
        "declaration, search, 2 x proposal+terminal"
    );
    assert_eq!(counts.visited_count, 2);
    assert_eq!(counts.candidate_count, 2);
    assert_eq!(counts.trial_count, 2);
    assert_eq!(counts.new_trial_count, 2);
    assert_eq!(counts.completed_count, 2);
    assert_eq!(counts.restored_trial_count, 0);
    assert_eq!(outcome.attempts.len(), 2);
    assert_eq!(
        outcome.attempts[1].seed_resolutions[0].first_actual_use_id, "choice:1",
        "a reused seed names its first use inside this attempt"
    );
}

#[test]
fn immediate_exhaustion_completes_with_no_trials() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Exhausted]);
    let mut buffer = Vec::new();
    let outcome = run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap();
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Exhausted);
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(counts.scope_count, 2);
    assert_eq!(counts.trial_count, 0);
}

#[test]
fn owner_cancellation_completes_at_a_trial_boundary() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut progress = Progress {
        cancel_after: Some(1),
        ..Progress::default()
    };
    let resume = options(None);
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &request,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Cancelled);
    assert_eq!(source.asks, vec![0]);
    assert_eq!(verified_counts(&buffer, &outcome).completed_count, 1);
}

#[test]
fn deterministic_proposal_creates_no_seed_resolution() {
    let request = budget("capture:first", 1);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    source.mode = SeedMode::Deterministic;
    let mut buffer = Vec::new();
    let outcome = run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap();
    assert!(outcome.attempts[0].seed_resolutions.is_empty());
    assert_eq!(verified_counts(&buffer, &outcome).completed_count, 1);
}

#[test]
fn requested_zero_stays_distinct_from_an_absent_seed() {
    let mut request = budget("capture:first", 1);
    request.capture.seed_declarations[0].requested = Some(SeedValue::U64 { value: 0 });
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    source.mode = SeedMode::Explicit(0);
    let mut buffer = Vec::new();
    let outcome = run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap();
    assert_eq!(verified_counts(&buffer, &outcome).completed_count, 1);

    // The same resolution refuses against an absent request.
    let request = budget("capture:second", 1);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    source.mode = SeedMode::Explicit(0);
    let mut buffer = Vec::new();
    let abort =
        report(run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap_err());
    assert!(abort.primary_error.message.contains("seed_resolution"));
    assert_eq!(
        abort.failed_attempt.failure_stage,
        Some(FailureStage::ProposalConversion)
    );
    assert_eq!(source.transitions, vec![owner_ref(0)]);
    assert_no_completion(&buffer);
}

#[test]
fn derived_parent_chain_records_and_restores() {
    let mut first = budget("capture:first", 2);
    first.capture.seed_declarations = vec![declared("seed:root"), declared("seed:child")];
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    source.mode = SeedMode::Derived;
    let mut progress = Progress::default();
    let resume = options(None);
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &first,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    verified_counts(&buffer, &outcome);
    assert_eq!(
        outcome.attempts[1]
            .seed_resolutions
            .iter()
            .map(|seed| seed.seed_id.as_str())
            .collect::<Vec<_>>(),
        vec!["seed:root", "seed:child"],
        "parents replay before the seed derived from them"
    );

    let mut second = captured("capture:second", Vec::new());
    second.capture.seed_declarations = first.capture.seed_declarations.clone();
    let publication = outcome.publication.unwrap();
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(
        &publication,
        vec![Step::Exhausted, Step::Exhausted, Step::Propose(3)],
    );
    source.mode = SeedMode::Derived;
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let resumed = run_serial(
        &second,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    let counts = verified_counts(&buffer, &resumed);
    assert_eq!(counts.restored_trial_count, 2);
    assert_eq!(counts.new_trial_count, 1);
    assert_eq!(counts.completed_count, 3);
    assert_eq!(source.asks, vec![2]);
}

#[test]
fn every_named_stage_aborts_without_completion_and_keeps_its_cause() {
    let confirmed = || OwnedAttempt {
        owner_ref_hex: Some(owner_ref(0)),
        ownership: Ownership::Confirmed,
        status: Some(TrialStatus::Running),
        cleanup: CleanupState::NotAttempted,
        diagnostic_hex: None,
    };
    let cases = [
        (Step::RaiseAsk, None, FailureStage::Ask, 0),
        (
            Step::Fail {
                stage: FailureStage::ParameterResolution,
                owned: Some(confirmed()),
            },
            None,
            FailureStage::ParameterResolution,
            1,
        ),
        (
            Step::Fail {
                stage: FailureStage::ProposalConversion,
                owned: Some(confirmed()),
            },
            None,
            FailureStage::ProposalConversion,
            1,
        ),
        (Step::Unplannable, None, FailureStage::CandidatePlanning, 1),
        (Step::Propose(1), Some(TRIAL_0), FailureStage::Evaluation, 1),
    ];
    for (step, failing, stage, cleaned) in cases {
        let request = captured("capture:first", Vec::new());
        let mut source = Source::scripted(vec![step]);
        let mut buffer = Vec::new();
        let abort = report(
            run_serial(&request, &mut source, failing, Box::new(&mut buffer), None).unwrap_err(),
        );
        assert_eq!(abort.disposition, HostHpoCaptureDisposition::Aborted);
        assert_eq!(abort.failed_attempt.failure_stage, Some(stage));
        assert_eq!(source.transitions.len(), cleaned);
        assert!(source.tells.is_empty(), "no score for a failed trial");
        assert_eq!(
            abort.evidence.refusal_code.as_deref(),
            Some("incomplete_error"),
            "the flushed OperationFailed refuses finish"
        );
        assert_eq!(abort.evidence.flush, HostHpoCaptureFlush::Flushed);
        assert!(!buffer.is_empty(), "the failure prefix reached the sink");
        assert_no_completion(&buffer);
        assert_closed_wire(&abort.failed_attempt);
    }
}

#[test]
fn factory_failures_keep_their_actual_stage() {
    let request = captured("capture:first", Vec::new());
    let invocations = Arc::new(AtomicUsize::new(0));
    let registry = controllers(None, &invocations).unwrap();
    let provider = InMemoryDataProvider::new(ControllerId::new("controller:data").unwrap());
    let refused_provider = ProviderFactory { fail: true };
    let mut refused_controller = ControllerFactory::new(None);
    refused_controller.fail = true;
    let working_provider = ProviderFactory { fail: false };
    let cases: [(
        &dyn HostHpoCandidateProviderFactory,
        Option<&dyn HostHpoCandidateControllerFactory>,
        FailureStage,
    ); 2] = [
        (&refused_provider, None, FailureStage::ProviderCreation),
        (
            &working_provider,
            Some(&refused_controller),
            FailureStage::ControllerCreation,
        ),
    ];
    for (provider_factory, controller_factory, stage) in cases {
        let mut source = Source::scripted(vec![Step::Propose(1)]);
        let mut buffer = Vec::new();
        let abort = report(
            SequentialScheduler
                .execute_captured_host_hpo_search(
                    &plan(),
                    &registry,
                    &provider,
                    Some(provider_factory),
                    controller_factory,
                    &request,
                    &mut source,
                    Box::new(&mut buffer),
                    None,
                )
                .unwrap_err(),
        );
        assert_eq!(abort.failed_attempt.failure_stage, Some(stage));
        assert_eq!(source.transitions, vec![owner_ref(0)]);
        assert_eq!(invocations.load(Ordering::SeqCst), 0);
        assert_no_completion(&buffer);
    }
}

#[test]
fn asks_five_six_seven_with_planning_failure_on_seven_fail_all_three() {
    let mut request = budget("capture:first", 8);
    request.request.phase_trial_budgets = vec![5, 3];
    let mut steps = (0..7).map(Step::Propose).collect::<Vec<_>>();
    steps.push(Step::Unplannable);
    let mut source = Source::scripted(steps);
    let mut progress = Progress::default();
    let factory = ControllerFactory::new(None);
    let resume = options(None);
    let mut buffer = Vec::new();
    let abort = report(
        run_parallel(
            &request,
            &mut source,
            &factory,
            Box::new(&mut buffer),
            3,
            Some((&resume, &mut progress)),
        )
        .unwrap_err(),
    );
    assert_eq!(source.asks, (0..8).collect::<Vec<_>>());
    assert_eq!(source.tells.len(), 5, "only the completed prefix was told");
    assert_eq!(
        source.transitions,
        vec![owner_ref(5), owner_ref(6), owner_ref(7)]
    );
    assert_eq!(
        progress.last().checkpoint.attempts.len(),
        5,
        "no window seal"
    );
    assert_eq!(abort.disposition, HostHpoCaptureDisposition::Aborted);
    let failed = &abort.failed_attempt;
    assert_eq!(failed.failure_stage, Some(FailureStage::CandidatePlanning));
    assert_eq!(failed.owned_attempts.len(), 3);
    for sibling in &failed.owned_attempts[..2] {
        assert_eq!(
            sibling.diagnostic_hex.as_deref(),
            Some(PLANNING_SIBLING_HEX)
        );
        assert_eq!(sibling.cleanup, CleanupState::Confirmed);
    }
    assert_ne!(
        failed.owned_attempts[2].diagnostic_hex.as_deref(),
        Some(PLANNING_SIBLING_HEX),
        "the planning failure keeps its actual cause"
    );
    assert_closed_wire(failed);
    assert_no_completion(&buffer);

    // The owner store moved on, so the old prefix no longer pairs.
    let last = progress.last();
    let mut second = captured("capture:second", Vec::new());
    second.request = request.request.clone();
    let resume = options(Some(last));
    let mut source = Source::resumed(vec![Step::Propose(1)], source.store.clone());
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let error = run_parallel(
        &second,
        &mut source,
        &factory,
        Box::new(&mut buffer),
        3,
        Some((&resume, &mut progress)),
    )
    .unwrap_err();
    assert_eq!(refusal_code(&error), "pair_mismatch");
    assert!(source.asks.is_empty() && source.fails.is_empty());
    assert!(progress.published.is_empty() && buffer.is_empty());
}

#[test]
fn raised_ask_keeps_unknown_ownership_and_cleans_only_confirmed_siblings() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2), Step::RaiseAsk]);
    let factory = ControllerFactory::new(None);
    let mut buffer = Vec::new();
    let abort = report(
        run_parallel(
            &request,
            &mut source,
            &factory,
            Box::new(&mut buffer),
            3,
            None,
        )
        .unwrap_err(),
    );
    assert_eq!(source.asks, vec![0, 1, 2]);
    assert_eq!(source.transitions, vec![owner_ref(0), owner_ref(1)]);
    assert!(
        source
            .fails
            .iter()
            .all(|(owner, ..)| *owner != owner_ref(2)),
        "a requested index never becomes a guessed owner ID"
    );
    assert!(
        source
            .fails
            .iter()
            .all(|(_, _, diagnostic)| diagnostic.contains("owner ask raised")),
        "the planning diagnostic stays off unrelated edges"
    );
    let failed = &abort.failed_attempt;
    assert_eq!(failed.failure_stage, Some(FailureStage::Ask));
    let unknown = failed
        .owned_attempts
        .iter()
        .filter(|attempt| attempt.ownership == Ownership::Unknown)
        .collect::<Vec<_>>();
    assert_eq!(unknown.len(), 1);
    assert_eq!(unknown[0].owner_ref_hex, None);
    assert_no_completion(&buffer);
}

#[test]
fn completed_and_pruned_trials_are_never_transitioned_again() {
    let mut request = captured("capture:first", Vec::new());
    request.request.progressive_pruning = true;
    request.request.fold_score_reduction = Some(HostHpoFoldReduction::Mean);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2), Step::Unplannable]);
    source.prune_on_report.insert(owner_ref(1));
    let mut buffer = Vec::new();
    let abort =
        report(run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap_err());
    assert_eq!(source.tells, vec![owner_ref(0)]);
    assert_eq!(source.pruned, vec![owner_ref(1)]);
    assert_eq!(source.fails.len(), 1, "only the live attempt is cleaned");
    assert_eq!(source.fails[0].0, owner_ref(2));
    assert_eq!(abort.failed_attempt.owned_attempts.len(), 1);
    assert_no_completion(&buffer);
}

#[test]
fn owner_terminal_observation_stays_open_in_the_stream_without_a_transition() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::ProposeTerminal(1, TrialStatus::Pruned)]);
    let mut buffer = Vec::new();
    let abort = report(
        run_serial(
            &request,
            &mut source,
            Some(TRIAL_0),
            Box::new(&mut buffer),
            None,
        )
        .unwrap_err(),
    );
    assert_eq!(source.fails.len(), 1, "status is observed once");
    assert!(
        source.transitions.is_empty(),
        "no second terminal transition"
    );
    let observed = &abort.failed_attempt.owned_attempts[0];
    assert_eq!(observed.status, Some(TrialStatus::Pruned));
    assert_eq!(observed.cleanup, CleanupState::NotAttempted);
    assert_no_completion(&buffer);
}

#[test]
fn cleanup_error_stays_beside_the_primary_failure() {
    let request = captured("capture:first", Vec::new());
    for durable in [false, true] {
        let mut source = Source::scripted(vec![Step::Propose(1)]);
        source.refuse_transition.insert(owner_ref(0));
        let mut progress = Progress::default();
        let resume = options(None);
        let mut buffer = Vec::new();
        let abort = report(
            run_serial(
                &request,
                &mut source,
                Some(TRIAL_0),
                Box::new(&mut buffer),
                durable.then_some((&resume, &mut progress)),
            )
            .unwrap_err(),
        );
        assert!(abort
            .primary_error
            .message
            .contains("deliberate evaluation"));
        assert!(abort
            .cleanup_errors
            .iter()
            .any(|error| error.message.contains("owner refused Failed transition")));
        assert_eq!(source.fails.len(), 1, "cleanup is attempted once");
        let observed = &abort.failed_attempt.owned_attempts[0];
        assert_eq!(observed.status, None, "the attempt stays uncertain");
        assert_eq!(observed.cleanup, CleanupState::Failed);
        assert_eq!(
            progress.seals(),
            0,
            "an unconfirmed Failed is not published"
        );
        assert_eq!(abort.disposition, HostHpoCaptureDisposition::Aborted);
        assert_no_completion(&buffer);
    }
}

#[test]
fn sink_failure_during_operation_failed_cannot_stop_cleanup_or_replace_the_primary() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let abort = report(
        run_serial(
            &request,
            &mut source,
            Some(TRIAL_0),
            Box::new(BrokenSink),
            None,
        )
        .unwrap_err(),
    );
    assert!(abort
        .primary_error
        .message
        .contains("deliberate evaluation"));
    assert_eq!(source.transitions, vec![owner_ref(0)], "cleanup still ran");
    assert_eq!(abort.evidence.refusal_code.as_deref(), Some("sink"));
    assert!(abort
        .evidence_errors
        .iter()
        .any(|error| error.message.contains("realization evidence refused")));
    assert!(
        abort.cleanup_errors.is_empty(),
        "evidence is a separate class"
    );
    assert_closed_wire(&abort.failed_attempt);
}

#[test]
fn sink_failure_at_finish_is_evidence_incomplete_with_no_commitment() {
    let request = budget("capture:first", 1);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    let abort =
        report(run_serial(&request, &mut source, None, Box::new(BrokenSink), None).unwrap_err());
    assert_eq!(
        abort.disposition,
        HostHpoCaptureDisposition::EvidenceIncomplete
    );
    assert_eq!(
        source.tells,
        vec![owner_ref(0)],
        "the search itself returned"
    );
    assert_eq!(abort.evidence.refusal_code.as_deref(), Some("sink"));
}

#[test]
fn unnamed_tell_failure_claims_no_false_stage() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    source.refuse_tell = true;
    let mut buffer = Vec::new();
    let abort =
        report(run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap_err());
    assert_eq!(abort.failed_attempt.failure_stage, None);
    assert!(abort.primary_error.message.contains("owner tell refused"));
    assert_eq!(source.fails[0].1, None);
    assert_eq!(
        abort.evidence.refusal_code.as_deref(),
        Some("empty_or_unclosed"),
        "no OperationFailed exists, so the Search stays open"
    );
    assert_no_completion(&buffer);
}

#[test]
fn conflicting_seed_value_across_proposals_refuses_before_the_recorder() {
    let request = budget("capture:first", 2);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    source.repeat_resolution = true;
    let mut buffer = Vec::new();
    let abort =
        report(run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap_err());
    assert!(abort
        .primary_error
        .message
        .contains("seed_resolution_conflict"));
    assert_eq!(
        abort.failed_attempt.failure_stage,
        Some(FailureStage::ProposalConversion),
        "the conflict is refused before it can poison the stream"
    );
    assert_eq!(
        abort.evidence.refusal_code.as_deref(),
        Some("incomplete_error"),
        "the stream survives and records the OperationFailed"
    );
    assert!(abort.evidence_errors.is_empty());
    assert_eq!(source.tells, vec![owner_ref(0)]);
    assert_eq!(source.transitions, vec![owner_ref(1)], "cleanup still ran");
    assert_no_completion(&buffer);
}

#[test]
fn published_failed_prefix_resumes_with_a_real_pair_and_no_extra_draw() {
    let first = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut progress = Progress::default();
    let resume = options(None);
    let mut buffer = Vec::new();
    let abort = report(
        run_serial(
            &first,
            &mut source,
            Some(TRIAL_1),
            Box::new(&mut buffer),
            Some((&resume, &mut progress)),
        )
        .unwrap_err(),
    );
    assert_eq!(
        abort.disposition,
        HostHpoCaptureDisposition::PublishedFailedPrefix
    );
    assert_eq!(source.transitions, vec![owner_ref(1)]);
    assert_no_completion(&buffer);
    let (publication, status) = progress.published.last().unwrap().clone();
    assert_eq!(status, HostHpoSearchStatus::Failed);
    assert!(publication.pair_fingerprint.is_some(), "a real pair");
    let attempt = &publication.checkpoint.attempts[1];
    assert_eq!(attempt.failure_stage, Some(FailureStage::Evaluation));
    assert_eq!(
        attempt.candidate.as_ref().unwrap().owner_variant_id,
        TRIAL_1,
        "identity comes from the actual candidate plan"
    );
    let prefix = abort.failed_attempt.trusted_prefix_ref.clone().unwrap();
    assert_eq!(
        prefix.checkpoint_fingerprint,
        publication.checkpoint.fingerprint
    );
    assert_eq!(prefix.pair_fingerprint, publication.pair_fingerprint);

    // Provenance comes from the sealed prefix, not the lost abort report.
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();
    assert_closed_wire(&derived[0]);
    let second = captured("capture:second", derived);
    let resume = options(Some(publication.clone()));
    let mut source = Source::resumed(
        vec![Step::Exhausted, Step::Exhausted, Step::Propose(3)],
        source.store.clone(),
    );
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &second,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(source.asks, vec![2], "restoration makes no new draw");
    assert!(source.fails.is_empty());
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Completed);
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(counts.restored_trial_count, 2);
    assert_eq!(counts.resume_observation_count, 2);
    assert_eq!(counts.new_trial_count, 1);
    assert_eq!(counts.failed_count, 1);
    assert_eq!(outcome.attempts[1].capture_id, "capture:first");
    assert_eq!(outcome.attempts[2].capture_id, "capture:second");
    let last = outcome.publication.unwrap();
    assert_eq!(
        last.checkpoint.attempts[..2],
        publication.checkpoint.attempts[..]
    );
}

#[test]
fn pending_attempts_replay_from_sealed_values_with_no_owner_ask() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2), Step::Propose(3)]);
    let mut progress = Progress::default();
    let factory = ControllerFactory::new(None);
    let resume = options(None);
    let mut buffer = Vec::new();
    run_parallel(
        &request,
        &mut source,
        &factory,
        Box::new(&mut buffer),
        3,
        Some((&resume, &mut progress)),
    )
    .unwrap();

    // The publication after trial 0 is what a crash would leave behind.
    let (publication, _) = progress
        .published
        .iter()
        .find(|(publication, _)| publication.checkpoint.attempts.len() == 1)
        .unwrap()
        .clone();
    assert_eq!(publication.checkpoint.pending_attempts.len(), 2);

    let second = captured("capture:second", Vec::new());
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = Progress::default();
    let factory = ControllerFactory::new(None);
    let mut buffer = Vec::new();
    let outcome = run_parallel(
        &second,
        &mut source,
        &factory,
        Box::new(&mut buffer),
        3,
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert!(
        source.asks.is_empty(),
        "pending replay never reaches the owner"
    );
    assert_eq!(source.tells, vec![owner_ref(1), owner_ref(2)]);
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(counts.restored_trial_count, 3);
    assert_eq!(counts.new_trial_count, 0);
    assert_eq!(counts.completed_count, 3);
}

#[test]
fn resume_refuses_before_any_ask_seal_or_recovery() {
    let first = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    let mut progress = Progress::default();
    let resume = options(None);
    let abort = report(
        run_serial(
            &first,
            &mut source,
            Some(TRIAL_0),
            Box::new(std::io::sink()),
            Some((&resume, &mut progress)),
        )
        .unwrap_err(),
    );
    assert_closed_wire(&abort.failed_attempt);
    let publication = progress.last();
    let store = source.store.clone();
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();

    let mut wrong_cause = derived[0].clone();
    wrong_cause.causal_diagnostic_hex = "deadbeef".into();
    let mut mutated = store.clone();
    mutated.insert("ff".into(), TrialStatus::Failed);

    let mut null_pair = publication.clone();
    null_pair.pair_fingerprint = None;
    null_pair.owner_checkpoint_fingerprint = None;
    let mut forged_pair = publication.clone();
    forged_pair.pair_fingerprint = Some("7".repeat(64));

    let cases = [
        (
            "omitted",
            publication.clone(),
            Vec::new(),
            store.clone(),
            "unpaired_failed_trial",
        ),
        (
            "cause",
            publication.clone(),
            vec![wrong_cause],
            store.clone(),
            "unpaired_failed_trial",
        ),
        (
            "store",
            publication.clone(),
            derived.clone(),
            mutated,
            "pair_mismatch",
        ),
        (
            "null",
            null_pair,
            derived.clone(),
            store.clone(),
            "pair_unavailable",
        ),
        (
            "forged",
            forged_pair,
            derived.clone(),
            store.clone(),
            "pair_mismatch",
        ),
    ];
    for (label, restored, failed_attempts, store, code) in cases {
        let second = captured("capture:second", failed_attempts);
        let resume = options(Some(restored));
        let mut source = Source::resumed(vec![Step::Propose(1), Step::Propose(2)], store);
        let mut progress = Progress::default();
        let mut buffer = Vec::new();
        let error = run_serial(
            &second,
            &mut source,
            None,
            Box::new(&mut buffer),
            Some((&resume, &mut progress)),
        )
        .unwrap_err();
        assert_eq!(refusal_code(&error), code, "{label}");
        assert!(source.asks.is_empty(), "{label}: no ask");
        assert!(source.fails.is_empty(), "{label}: no recovery");
        assert!(progress.published.is_empty(), "{label}: no seal");
        assert!(buffer.is_empty(), "{label}: no stream");
    }
}

#[test]
fn owner_row_count_must_match_the_publication() {
    let first = budget("capture:first", 2);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    source.pinned_fingerprint = Some("a".repeat(64));
    let mut progress = Progress::default();
    let resume = options(None);
    let outcome = run_serial(
        &first,
        &mut source,
        None,
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Exhausted);
    let publication = outcome.publication.unwrap();
    let second = budget("capture:second", 2);
    let resume = options(Some(publication));
    let mut store = source.store.clone();
    store.insert("ff".into(), TrialStatus::Running);
    let mut source = Source::resumed(vec![Step::Propose(1)], store);
    source.pinned_fingerprint = Some("a".repeat(64));
    let mut progress = Progress::default();
    let error = run_serial(
        &second,
        &mut source,
        None,
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap_err();
    assert_eq!(refusal_code(&error), "owner_row_mismatch");
    assert!(source.asks.is_empty());
}

#[test]
fn null_owner_state_publishes_an_honest_null_pair_that_cannot_resume() {
    let first = budget("capture:first", 1);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    source.state_unavailable = true;
    let mut progress = Progress::default();
    let resume = options(None);
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &first,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    verified_counts(&buffer, &outcome);
    let publication = outcome.publication.unwrap();
    assert_eq!(publication.pair_fingerprint, None);
    let second = budget("capture:second", 1);
    let resume = options(Some(publication));
    let mut source = Source::resumed(Vec::new(), source.store.clone());
    let mut progress = Progress::default();
    let error = run_serial(
        &second,
        &mut source,
        None,
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap_err();
    assert_eq!(refusal_code(&error), "pair_unavailable");
}

#[test]
fn fresh_start_requires_an_empty_owner_store() {
    let request = captured("capture:first", Vec::new());
    let store = BTreeMap::from([("a0".to_string(), TrialStatus::Failed)]);
    let mut source = Source::resumed(vec![Step::Propose(1)], store);
    let error =
        run_serial(&request, &mut source, None, Box::new(std::io::sink()), None).unwrap_err();
    assert_eq!(refusal_code(&error), "fresh_store_required");
    assert!(source.asks.is_empty());

    let confirmed = FailedAttempt {
        failed_capture_id: Some("capture:earlier".into()),
        failed_scope_id: Some(SCOPE.into()),
        failure_stage: Some(FailureStage::CandidatePlanning),
        causal_diagnostic_hex: hex("planning"),
        owned_attempts: vec![OwnedAttempt {
            owner_ref_hex: Some("a0".into()),
            ownership: Ownership::Confirmed,
            status: Some(TrialStatus::Failed),
            cleanup: CleanupState::Confirmed,
            diagnostic_hex: None,
        }],
        trusted_prefix_ref: None,
    };
    let request = captured("capture:first", vec![confirmed]);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    let error =
        run_serial(&request, &mut source, None, Box::new(std::io::sink()), None).unwrap_err();
    assert_eq!(refusal_code(&error), "fresh_store_required");
    assert!(source.asks.is_empty());
}

#[test]
fn worker_window_planning_failure_cleans_every_pending_ask() {
    let request = budget("capture:first", 2);
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Unplannable]);
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let abort = report(
        session
            .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
            .unwrap_err(),
    );
    assert_eq!(source.asks, vec![0, 1]);
    assert_eq!(source.transitions, vec![owner_ref(0), owner_ref(1)]);
    assert_eq!(
        abort.failed_attempt.owned_attempts[0]
            .diagnostic_hex
            .as_deref(),
        Some(PLANNING_SIBLING_HEX)
    );
    assert!(
        session
            .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
            .is_err(),
        "an aborted session cannot open another window"
    );
    assert_eq!(source.asks, vec![0, 1], "no owner call after the abort");
    drop(session);
    assert_no_completion(&buffer);
}

#[test]
fn worker_window_returned_failed_completes_and_resumes_from_derived_provenance() {
    let request = captured("capture:first", Vec::new());
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2), Step::Propose(3)]);
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    assert!(
        session
            .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
            .is_err(),
        "an open window cannot be reused"
    );
    assert_eq!(source.asks, vec![0, 1], "the second prepare asks nothing");

    let invocations = Arc::new(AtomicUsize::new(0));
    let registry = controllers(None, &invocations).unwrap();
    let provider = InMemoryDataProvider::new(ControllerId::new("controller:data").unwrap());
    let evidence =
        evaluate_host_hpo_worker_task(&window.tasks[0], &request.request, &registry, &provider)
            .unwrap();
    let results = vec![
        HostHpoWorkerResult::Complete {
            evidence,
            data_fingerprint: DATA.into(),
            fold_evidence: Vec::new(),
        },
        HostHpoWorkerResult::Failed {
            trial_index: 1,
            error: "worker failed".into(),
        },
    ];
    let mut progress = Progress::default();
    let outcome = session
        .complete_worker_window(
            &plan,
            &request,
            &resume,
            window,
            results,
            BTreeMap::from([(1, FailureStage::Evaluation)]),
            &mut source,
            &mut progress,
        )
        .unwrap();
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Failed);
    assert_eq!(
        outcome.failed_attempts.len(),
        1,
        "provenance survives return"
    );
    let publication = outcome.publication.clone().unwrap();

    // A terminal session makes no owner call, even with its own publication.
    let current = options(Some(publication.clone()));
    assert!(session
        .prepare_worker_window(&plan, &request, &current, &mut source, 2)
        .is_err());
    assert_eq!(source.asks, vec![0, 1], "no ask after a terminal status");
    drop(session);
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(counts.completed_count, 1);
    assert_eq!(counts.failed_count, 1);

    let second = captured("capture:second", outcome.failed_attempts.clone());
    let mut resumed_source = Source::resumed(
        vec![Step::Exhausted, Step::Exhausted, Step::Propose(3)],
        source.store.clone(),
    );
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let resumed = run_serial(
        &second,
        &mut resumed_source,
        None,
        Box::new(&mut buffer),
        Some((&current, &mut progress)),
    )
    .unwrap();
    assert_eq!(resumed_source.asks, vec![2]);
    assert_eq!(verified_counts(&buffer, &resumed).trial_count, 3);
}

#[derive(Default)]
struct OrdinaryProposals {
    asks: usize,
    fails: usize,
    empty: bool,
}

impl HostHpoProposalSource for OrdinaryProposals {
    fn ask(&mut self, _: u32) -> Result<Option<BTreeMap<String, Value>>> {
        self.asks += 1;
        Ok(Some(if self.empty {
            BTreeMap::new()
        } else {
            BTreeMap::from([("n_components".to_string(), json!(1))])
        }))
    }
    fn tell(&mut self, _: u32, _: f64) -> Result<()> {
        Ok(())
    }
    fn fail(&mut self, _: u32, _: &str) -> Result<()> {
        self.fails += 1;
        Ok(())
    }
}

#[test]
fn noncapture_route_keeps_its_untyped_errors_and_cleanup_behavior() {
    let invocations = Arc::new(AtomicUsize::new(0));
    let provider = InMemoryDataProvider::new(ControllerId::new("controller:data").unwrap());
    for (failing, empty, expected) in [
        (Some(TRIAL_0), false, "deliberate evaluation failure"),
        (None, true, "host HPO proposal parameters must be nonempty"),
    ] {
        let registry = controllers(failing, &invocations).unwrap();
        let mut proposals = OrdinaryProposals {
            empty,
            ..OrdinaryProposals::default()
        };
        let error = SequentialScheduler
            .execute_host_hpo_search(
                &plan(),
                &registry,
                &provider,
                &search_request(),
                &mut proposals,
            )
            .unwrap_err();
        assert!(matches!(error, DagMlError::RuntimeValidation(_)), "{error}");
        assert!(error.to_string().contains(expected), "{error}");
        assert_eq!(proposals.asks, 1);
        assert_eq!(proposals.fails, 0, "serial non-durable v1 sends no fail");
    }
}

#[test]
fn undeclared_plan_node_refuses_before_any_ask() {
    let mut request = captured("capture:first", Vec::new());
    request.capture.node_ids = vec!["model:pls".into()];
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    let error =
        run_serial(&request, &mut source, None, Box::new(std::io::sink()), None).unwrap_err();
    assert!(
        error.to_string().contains("declaration_node_mismatch"),
        "{error}"
    );
    assert!(source.asks.is_empty());
}

/// The publication a crash leaves after trial 0 of one all-asked parallel window.
fn b2c_regression_pending_publication(trial_budget: u32) -> HostHpoCapturePublication {
    let request = budget("capture:first", trial_budget);
    let steps = (1..=i64::from(trial_budget)).map(Step::Propose).collect();
    let mut source = Source::scripted(steps);
    let mut progress = Progress::default();
    let factory = ControllerFactory::new(None);
    let resume = options(None);
    run_parallel(
        &request,
        &mut source,
        &factory,
        Box::new(std::io::sink()),
        trial_budget as usize,
        Some((&resume, &mut progress)),
    )
    .unwrap();
    progress
        .published
        .iter()
        .find(|(publication, _)| publication.checkpoint.attempts.len() == 1)
        .unwrap()
        .0
        .clone()
}

fn b2c_regression_complete_result(
    task: &HostHpoWorkerTask,
    request: &HostHpoCaptureSearchRequest,
) -> HostHpoWorkerResult {
    let invocations = Arc::new(AtomicUsize::new(0));
    let registry = controllers(None, &invocations).unwrap();
    let provider = InMemoryDataProvider::new(ControllerId::new("controller:data").unwrap());
    let evidence =
        evaluate_host_hpo_worker_task(task, &request.request, &registry, &provider).unwrap();
    HostHpoWorkerResult::Complete {
        evidence,
        data_fingerprint: DATA.into(),
        fold_evidence: Vec::new(),
    }
}

/// Pending trials outside the budget could never be served, so a completion
/// would silently omit trials the owner still holds as RUNNING.
fn b2c_regression_assert_pending_budget_refusal(trial_budget: u32) {
    let publication = b2c_regression_pending_publication(3);
    assert_eq!(publication.checkpoint.pending_attempts.len(), 2);
    let request = budget("capture:second", trial_budget);
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = Progress::default();
    let factory = ControllerFactory::new(None);
    let mut buffer = Vec::new();
    let result = run_parallel(
        &request,
        &mut source,
        &factory,
        Box::new(&mut buffer),
        3,
        Some((&resume, &mut progress)),
    );
    assert!(
        result.is_err(),
        "a pending index at or beyond trial_budget must refuse"
    );
    assert!(source.asks.is_empty(), "refusal precedes any ask");
    assert!(source.tells.is_empty(), "refusal precedes any transition");
    assert!(source.fails.is_empty(), "refusal precedes any cleanup");
    assert!(progress.published.is_empty(), "refusal precedes any seal");
    assert!(buffer.is_empty(), "refusal precedes the stream");
}

#[test]
fn b2c_regression_cancelled_resume_with_retained_pending_has_no_verified_completion() {
    let publication = b2c_regression_pending_publication(3);
    assert_eq!(publication.checkpoint.pending_attempts.len(), 2);
    let request = captured("capture:second", Vec::new());
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = Progress {
        cancel_after: Some(1),
        ..Progress::default()
    };
    let factory = ControllerFactory::new(None);
    let mut buffer = Vec::new();
    let result = run_parallel(
        &request,
        &mut source,
        &factory,
        Box::new(&mut buffer),
        3,
        Some((&resume, &mut progress)),
    );
    if let Ok(outcome) = &result {
        assert!(
            outcome.completion.is_none(),
            "a completion would omit two RUNNING owner trials"
        );
        assert_eq!(outcome.uncaptured_pending, vec![1, 2]);
        assert!(outcome.publication.is_some(), "the prefix stays published");
    }
    assert!(result.is_ok(), "cancellation itself is not a failure");
    assert_no_completion(&buffer);
    let restated = stream_events(&buffer)
        .into_iter()
        .filter(|event| event["kind"] == "trial_resume" && event["observed_status"] == "running")
        .count();
    assert_eq!(
        restated, 2,
        "the prefix shows the trials it could not close"
    );
    assert!(source.asks.is_empty());
    assert!(source.tells.is_empty());
    assert_eq!(source.store[&owner_ref(1)], TrialStatus::Running);
    assert_eq!(source.store[&owner_ref(2)], TrialStatus::Running);
}

#[test]
fn b2c_regression_every_pending_index_beyond_budget_refuses_before_any_ask() {
    b2c_regression_assert_pending_budget_refusal(1);
}

#[test]
fn b2c_regression_last_pending_index_equal_to_budget_refuses_before_any_ask() {
    b2c_regression_assert_pending_budget_refusal(2);
}

#[test]
fn b2c_regression_worker_failed_window_with_retained_pending_has_no_verified_completion() {
    let publication = b2c_regression_pending_publication(4);
    assert_eq!(publication.checkpoint.pending_attempts.len(), 3);
    let mut request = captured("capture:second", Vec::new());
    request.request.trial_budget = 4;
    let plan = plan();
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    assert_eq!(window.tasks.len(), 2, "trial 3 stays retained");
    let results = window
        .tasks
        .iter()
        .map(|task| HostHpoWorkerResult::Failed {
            trial_index: task.trial_index,
            error: "worker failed".into(),
        })
        .collect();
    let mut progress = Progress::default();
    let result = session.complete_worker_window(
        &plan,
        &request,
        &resume,
        window,
        results,
        BTreeMap::new(),
        &mut source,
        &mut progress,
    );
    drop(session);
    if let Ok(outcome) = &result {
        assert!(
            outcome.completion.is_none(),
            "a completion would omit a RUNNING owner trial"
        );
        assert_eq!(outcome.uncaptured_pending, vec![3]);
    }
    assert!(result.is_ok());
    assert_no_completion(&buffer);
    assert!(source.asks.is_empty());
    assert_eq!(source.store[&owner_ref(3)], TrialStatus::Running);
}

#[test]
fn b2c_regression_worker_empty_first_window_completes_with_a_verified_stream() {
    let request = captured("capture:first", Vec::new());
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Exhausted]);
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    assert!(window.tasks.is_empty());
    assert!(window.exhausted);
    let mut progress = Progress::default();
    let outcome = session
        .complete_worker_window(
            &plan,
            &request,
            &resume,
            window,
            Vec::new(),
            BTreeMap::new(),
            &mut source,
            &mut progress,
        )
        .expect("an empty exhausted window is a valid completion");
    drop(session);
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Exhausted);
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(counts.scope_count, 2);
    assert_eq!(counts.trial_count, 0);
    assert_eq!(source.asks, vec![0]);
}

#[test]
fn b2c_regression_worker_empty_later_window_completes_with_a_verified_stream() {
    let request = captured("capture:first", Vec::new());
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2), Step::Exhausted]);
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    let results = window
        .tasks
        .iter()
        .map(|task| b2c_regression_complete_result(task, &request))
        .collect();
    let mut progress = Progress::default();
    let running = session
        .complete_worker_window(
            &plan,
            &request,
            &resume,
            window,
            results,
            BTreeMap::new(),
            &mut source,
            &mut progress,
        )
        .unwrap();
    assert_eq!(running.outcome.status, HostHpoSearchStatus::Running);
    assert!(running.completion.is_none(), "the search is still open");

    let current = options(running.publication);
    let window = session
        .prepare_worker_window(&plan, &request, &current, &mut source, 2)
        .unwrap();
    assert!(window.tasks.is_empty());
    assert!(window.exhausted);
    let outcome = session
        .complete_worker_window(
            &plan,
            &request,
            &current,
            window,
            Vec::new(),
            BTreeMap::new(),
            &mut source,
            &mut progress,
        )
        .expect("a later empty exhausted window is a valid completion");
    drop(session);
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Exhausted);
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(counts.trial_count, 2);
    assert_eq!(counts.completed_count, 2);
    assert_eq!(source.asks, vec![0, 1, 2]);
}

#[test]
fn b2c_regression_worker_empty_window_at_budget_completes_with_a_verified_stream() {
    let first = budget("capture:first", 2);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut progress = Progress::default();
    let resume = options(None);
    let finished = run_serial(
        &first,
        &mut source,
        None,
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    let publication = finished.publication.unwrap();

    let request = budget("capture:second", 2);
    let plan = plan();
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    assert!(window.tasks.is_empty());
    let mut progress = Progress::default();
    let outcome = session
        .complete_worker_window(
            &plan,
            &request,
            &resume,
            window,
            Vec::new(),
            BTreeMap::new(),
            &mut source,
            &mut progress,
        )
        .expect("a resume already at budget is a valid completion");
    drop(session);
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Completed);
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(counts.restored_trial_count, 2);
    assert_eq!(counts.new_trial_count, 0);
    assert!(source.asks.is_empty());
}

#[test]
fn b2c_regression_stageless_worker_failure_is_not_reported_as_evaluation() {
    let request = captured("capture:first", Vec::new());
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    let results = vec![
        b2c_regression_complete_result(&window.tasks[0], &request),
        HostHpoWorkerResult::Failed {
            trial_index: 1,
            error: "provider creation failed: invalid envelope".into(),
        },
    ];
    let mut progress = Progress::default();
    let outcome = session
        .complete_worker_window(
            &plan,
            &request,
            &resume,
            window,
            results,
            BTreeMap::new(),
            &mut source,
            &mut progress,
        )
        .unwrap();
    drop(session);
    assert_eq!(
        source.fails[0].1, None,
        "a worker failure carries no stage, so none may be claimed"
    );
    assert_eq!(outcome.attempts[1].failure_stage, None);
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Failed);
    assert_eq!(verified_counts(&buffer, &outcome).failed_count, 1);
    let events = stream_events(&buffer);
    assert!(
        events
            .iter()
            .all(|event| event["kind"] != "operation_failed"),
        "a returned Failed is not an operation failure"
    );
    let outcomes = events
        .iter()
        .filter(|event| event["kind"] == "trial_outcome" && event["outcome"]["status"] == "failed")
        .collect::<Vec<_>>();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0]["outcome"]["stage"]["availability"], "unknown");

    // Qualified resume requires the actual stage, so this prefix cannot resume.
    let publication = outcome.publication.unwrap();
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();
    let second = captured("capture:second", derived);
    let resume = options(Some(publication));
    let mut resumed = Source::resumed(vec![Step::Propose(3)], source.store.clone());
    let mut progress = Progress::default();
    let error = run_serial(
        &second,
        &mut resumed,
        None,
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap_err();
    assert_eq!(refusal_code(&error), "attempt_terminal");
    assert!(resumed.asks.is_empty());
}

/// A first capture that publishes a Failed trial 0, and its owner store.
fn b2c_published_failure(
    owner_diagnostic: Option<String>,
) -> (HostHpoCapturePublication, BTreeMap<String, TrialStatus>) {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    source.owner_diagnostic = owner_diagnostic;
    let mut progress = Progress::default();
    let resume = options(None);
    run_serial(
        &request,
        &mut source,
        Some(TRIAL_0),
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap_err();
    (progress.last(), source.store)
}

#[test]
fn b2c_replayed_failure_is_attributed_to_the_capture_that_observed_it() {
    let publication = b2c_regression_pending_publication(3);
    let second = captured("capture:second", Vec::new());
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = Progress::default();
    let factory = ControllerFactory::new(Some(TRIAL_1));
    let abort = report(
        run_parallel(
            &second,
            &mut source,
            &factory,
            Box::new(std::io::sink()),
            3,
            Some((&resume, &mut progress)),
        )
        .unwrap_err(),
    );
    assert_eq!(
        abort.failed_attempt.failed_capture_id.as_deref(),
        Some("capture:second")
    );
    assert!(source.asks.is_empty());
    let last = progress.last();
    let attempt = &last.checkpoint.attempts[1];
    assert_eq!(
        attempt.capture_id, "capture:first",
        "the ask stays original"
    );
    assert_eq!(
        attempt.terminal_capture_id.as_deref(),
        Some("capture:second")
    );
    let derived = host_hpo_capture_prefix_provenance(&last).unwrap();
    assert_eq!(
        derived[0].failed_capture_id.as_deref(),
        Some("capture:second")
    );

    // A capture that observed a terminal cannot be reused, even with no ask.
    let reused = captured("capture:second", Vec::new());
    let resume = options(Some(last.clone()));
    let mut owner = Source::resumed(Vec::new(), source.store.clone());
    let mut progress = Progress::default();
    let error = run_serial(
        &reused,
        &mut owner,
        None,
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap_err();
    assert_eq!(refusal_code(&error), "attempt_identity");

    let third = captured("capture:third", derived);
    let mut owner = Source::resumed(Vec::new(), source.store.clone());
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &third,
        &mut owner,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(verified_counts(&buffer, &outcome).restored_trial_count, 3);
}

#[test]
fn b2c_truncated_owner_diagnostic_is_preserved_and_still_resumes() {
    let truncated = hex("deliberate eval");
    let (publication, store) = b2c_published_failure(Some(truncated.clone()));
    let attempt = &publication.checkpoint.attempts[0];
    assert_eq!(
        attempt.owner_observation.diagnostic_hex.as_deref(),
        Some(truncated.as_str()),
        "the owner's own text is kept as observed"
    );
    assert_ne!(
        attempt.causal_diagnostic_hex.as_deref(),
        Some(truncated.as_str())
    );
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();
    let second = captured("capture:second", derived);
    let resume = options(Some(publication));
    let mut source = Source::resumed(
        vec![Step::Exhausted, Step::Propose(2), Step::Propose(3)],
        store,
    );
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &second,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(source.asks, vec![1, 2]);
    assert_eq!(verified_counts(&buffer, &outcome).restored_trial_count, 1);
}

#[test]
fn b2c_cleanup_beyond_a_published_failure_is_an_abort_not_a_resumable_prefix() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2), Step::Propose(3)]);
    source.refuse_tell = true;
    let mut progress = Progress::default();
    let factory = ControllerFactory::new(Some(TRIAL_0));
    let resume = options(None);
    let abort = report(
        run_parallel(
            &request,
            &mut source,
            &factory,
            Box::new(std::io::sink()),
            3,
            Some((&resume, &mut progress)),
        )
        .unwrap_err(),
    );
    assert_eq!(progress.seals(), 1, "trial 0 did publish a Failed terminal");
    assert_eq!(
        source.transitions,
        vec![owner_ref(0), owner_ref(1), owner_ref(2)],
        "cleanup moved the owner beyond that publication"
    );
    assert_eq!(abort.disposition, HostHpoCaptureDisposition::Aborted);
    assert!(abort
        .secondary_errors
        .iter()
        .any(|error| error.message.contains("owner tell refused")));
}

#[test]
fn b2c_restored_seed_facts_name_the_scope_that_recorded_them() {
    let first = budget("capture:first", 1);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    let mut progress = Progress::default();
    let resume = options(None);
    let finished = run_serial(
        &first,
        &mut source,
        None,
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    let publication = finished.publication.unwrap();
    assert_eq!(publication.checkpoint.attempts[0].search_scope_id, SEARCH);
    let second = budget("capture:second", 1);
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    run_serial(
        &second,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    let events = stream_events(&buffer);
    let original_scope = |kind: &str| {
        events
            .iter()
            .find(|event| event["kind"] == kind && event["provenance"]["observation"] == "restored")
            .map(|event| event["provenance"]["original_scope_id"].clone())
            .unwrap()
    };
    assert_eq!(original_scope("seed_declared"), SEARCH);
    assert_eq!(original_scope("effective_seed_resolved"), SEARCH);
    assert_eq!(original_scope("choice"), "proposal:0000000000");
}

#[test]
fn b2c_returned_failed_evidence_is_the_actual_cause() {
    let request = captured("capture:first", Vec::new());
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    let results = vec![
        b2c_regression_complete_result(&window.tasks[0], &request),
        HostHpoWorkerResult::Failed {
            trial_index: 1,
            error: "worker failed".into(),
        },
    ];
    let mut progress = Progress::default();
    session
        .complete_worker_window(
            &plan,
            &request,
            &resume,
            window,
            results,
            BTreeMap::new(),
            &mut source,
            &mut progress,
        )
        .unwrap();
    drop(session);
    let events = stream_events(&buffer);
    let close = events
        .iter()
        .find(|event| event["kind"] == "search_close")
        .unwrap();
    assert_eq!(close["evidence"]["value_hex"], hex("worker failed"));
}

#[test]
fn b2c_owner_state_must_be_a_canonical_digest_before_any_ask() {
    for digest in ["", "owner-state", "ABCDEF"] {
        let request = captured("capture:first", Vec::new());
        let mut source = Source::scripted(vec![Step::Propose(1)]);
        source.pinned_fingerprint = Some(digest.to_owned());
        let error =
            run_serial(&request, &mut source, None, Box::new(std::io::sink()), None).unwrap_err();
        assert_eq!(refusal_code(&error), "owner_state_invalid", "{digest}");
        assert!(source.asks.is_empty());
    }
}

#[test]
fn b2c_owner_digest_is_state_sensitive_and_rng_drift_with_equal_rows_refuses() {
    let source = Source::scripted(Vec::new());
    let digest = source.state_digest();
    assert_eq!(digest.len(), 64);
    let mut moved = Source::scripted(Vec::new());
    moved.rng_state = 1;
    assert_ne!(
        moved.state_digest(),
        digest,
        "RNG position changes the digest"
    );

    let publication = b2c_regression_pending_publication(3);
    let mut owner = Source::at_publication(&publication, Vec::new());
    let unchanged = owner.state_digest();
    assert_eq!(
        Some(unchanged.as_str()),
        publication.owner_checkpoint_fingerprint.as_deref()
    );
    owner.trial_params.insert(owner_ref(1), 99);
    assert_ne!(owner.state_digest(), unchanged, "pending params change it");

    // The rows match exactly; only the sampler moved.
    let second = captured("capture:second", Vec::new());
    let resume = options(Some(publication.clone()));
    let mut drifted = Source::at_publication(&publication, Vec::new());
    drifted.rng_state += 1;
    let mut progress = Progress::default();
    let factory = ControllerFactory::new(None);
    let error = run_parallel(
        &second,
        &mut drifted,
        &factory,
        Box::new(std::io::sink()),
        3,
        Some((&resume, &mut progress)),
    )
    .unwrap_err();
    assert_eq!(refusal_code(&error), "pair_mismatch");
    assert!(drifted.asks.is_empty());
}

#[test]
fn b2c_worker_stage_is_recorded_and_resumes_when_actually_known() {
    let request = captured("capture:first", Vec::new());
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    let results = vec![
        b2c_regression_complete_result(&window.tasks[0], &request),
        HostHpoWorkerResult::Failed {
            trial_index: 1,
            error: "provider creation failed: invalid envelope".into(),
        },
    ];
    let mut progress = Progress::default();
    let outcome = session
        .complete_worker_window(
            &plan,
            &request,
            &resume,
            window,
            results,
            BTreeMap::from([(1, FailureStage::ProviderCreation)]),
            &mut source,
            &mut progress,
        )
        .unwrap();
    drop(session);
    assert_eq!(source.fails[0].1, Some(FailureStage::ProviderCreation));
    assert_eq!(
        outcome.attempts[1].failure_stage,
        Some(FailureStage::ProviderCreation)
    );
    verified_counts(&buffer, &outcome);

    let second = captured("capture:second", outcome.failed_attempts.clone());
    let resume = options(outcome.publication.clone());
    let mut resumed = Source::resumed(
        vec![Step::Exhausted, Step::Exhausted, Step::Propose(3)],
        source.store.clone(),
    );
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let finished = run_serial(
        &second,
        &mut resumed,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(resumed.asks, vec![2]);
    assert_eq!(verified_counts(&buffer, &finished).restored_trial_count, 2);
}

#[test]
fn b2c_invalid_worker_stages_refuse_with_no_owner_call() {
    let request = captured("capture:first", Vec::new());
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut buffer = Vec::new();
    let session = HostHpoCaptureSession::open(
        &plan,
        &request,
        Some(&resume),
        &mut source,
        Box::new(&mut buffer),
    )
    .unwrap();
    let window = session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .unwrap();
    let results = || {
        vec![
            b2c_regression_complete_result(&window.tasks[0], &request),
            HostHpoWorkerResult::Failed {
                trial_index: 1,
                error: "worker failed".into(),
            },
        ]
    };
    for stages in [
        BTreeMap::from([(0, FailureStage::Evaluation)]),
        BTreeMap::from([(1, FailureStage::CandidatePlanning)]),
        BTreeMap::from([(9, FailureStage::Evaluation)]),
    ] {
        let mut progress = Progress::default();
        assert!(session
            .complete_worker_window(
                &plan,
                &request,
                &resume,
                window.clone(),
                results(),
                stages,
                &mut source,
                &mut progress,
            )
            .is_err());
        assert!(progress.prepared.is_empty(), "nothing was prepared");
        assert!(progress.published.is_empty(), "nothing was published");
    }
    assert!(source.tells.is_empty() && source.fails.is_empty());

    // The refusal left the window open, so the honest call still works.
    let mut progress = Progress::default();
    let outcome = session
        .complete_worker_window(
            &plan,
            &request,
            &resume,
            window.clone(),
            results(),
            BTreeMap::from([(1, FailureStage::Evaluation)]),
            &mut source,
            &mut progress,
        )
        .unwrap();
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Failed);
}

#[test]
fn b2c_parallel_factory_failures_keep_their_actual_stage() {
    let request = captured("capture:first", Vec::new());
    let working = ControllerFactory::new(None);
    let mut refused = ControllerFactory::new(None);
    refused.fail = true;
    let cases: [(bool, &ControllerFactory, FailureStage); 2] = [
        (true, &working, FailureStage::ProviderCreation),
        (false, &refused, FailureStage::ControllerCreation),
    ];
    for (provider_fails, controller_factory, stage) in cases {
        let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
        let mut buffer = Vec::new();
        let abort = report(
            SequentialScheduler
                .execute_captured_parallel_host_hpo_search(
                    &plan(),
                    &ProviderFactory {
                        fail: provider_fails,
                    },
                    controller_factory,
                    &request,
                    &mut source,
                    Box::new(&mut buffer),
                    2,
                    None,
                )
                .unwrap_err(),
        );
        assert_eq!(abort.failed_attempt.failure_stage, Some(stage));
        assert_eq!(source.transitions, vec![owner_ref(0)]);
        assert_no_completion(&buffer);
    }
}

/// A first capture with a completed trial 0 and a published Failed trial 1.
fn b2d_failed_prefix() -> (HostHpoCapturePublication, Vec<FailedAttempt>) {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut progress = Progress::default();
    let resume = options(None);
    run_serial(
        &request,
        &mut source,
        Some(TRIAL_1),
        Box::new(std::io::sink()),
        Some((&resume, &mut progress)),
    )
    .unwrap_err();
    let publication = progress.last();
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();
    (publication, derived)
}

#[test]
fn b2d_persistent_owner_omits_a_restored_seed_and_resumes_with_no_draw() {
    let (publication, derived) = b2d_failed_prefix();
    let restored = publication.checkpoint.attempts[0].seed_resolutions[0].clone();
    let second = captured("capture:second", derived.clone());
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(
        &publication,
        vec![Step::Exhausted, Step::Exhausted, Step::Propose(3)],
    );
    source.seed_resolved = true; // The owner kept its state, as the pair requires.
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &second,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(source.asks, vec![2], "the seed fact cost no extra ask");
    assert!(source.fails.is_empty());
    let counts = verified_counts(&buffer, &outcome);
    assert_eq!(counts.restored_trial_count, 2);
    assert_eq!(counts.new_trial_count, 1);
    let carried = &outcome.attempts[2].seed_resolutions[0];
    assert_eq!(
        carried.effective, restored.effective,
        "no new value appears"
    );
    assert_eq!(
        carried.first_actual_use_id, "choice:2",
        "first use stays local"
    );
    let events = stream_events(&buffer);
    let live = events
        .iter()
        .filter(|event| {
            event["kind"] == "effective_seed_resolved"
                && event["scope_id"] == SEARCH
                && event["provenance"]["observation"] == "new"
        })
        .count();
    assert_eq!(live, 1, "the new lineage resolves the carried seed once");

    // The carried seed survives another lineage unchanged.
    let last = outcome.publication.unwrap();
    let third = captured(
        "capture:third",
        host_hpo_capture_prefix_provenance(&last).unwrap(),
    );
    let resume = options(Some(last.clone()));
    let mut owner = Source::at_publication(&last, Vec::new());
    owner.seed_resolved = true;
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let finished = run_serial(
        &third,
        &mut owner,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(verified_counts(&buffer, &finished).restored_trial_count, 3);
}

#[test]
fn b2d_persistent_owner_reusing_a_restored_seed_across_proposals_resolves_it_once() {
    let (publication, derived) = b2d_failed_prefix();
    let mut second = captured("capture:second", derived);
    second.request.trial_budget = 4;
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(
        &publication,
        vec![
            Step::Exhausted,
            Step::Exhausted,
            Step::Propose(3),
            Step::Propose(1),
        ],
    );
    source.seed_resolved = true;
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &second,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(source.asks, vec![2, 3]);
    assert_eq!(verified_counts(&buffer, &outcome).new_trial_count, 2);
    let resolved = stream_events(&buffer)
        .iter()
        .filter(|event| event["kind"] == "effective_seed_resolved" && event["scope_id"] == SEARCH)
        .count();
    assert_eq!(resolved, 1);
}

#[test]
fn b2d_serial_replay_keeps_citing_the_restored_checkpoint() {
    let publication = b2c_regression_pending_publication(3);
    let restored = publication.checkpoint.fingerprint.clone();
    let second = captured("capture:second", Vec::new());
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &second,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert!(source.asks.is_empty());
    assert!(
        progress
            .published
            .iter()
            .any(|(published, _)| published.checkpoint.fingerprint != restored),
        "trial 1 published its own checkpoint before trial 2 replayed"
    );
    assert_eq!(verified_counts(&buffer, &outcome).restored_trial_count, 3);
    let cited = stream_events(&buffer)
        .iter()
        .filter_map(|event| {
            event["provenance"]["checkpoint_ref"]
                .as_str()
                .map(str::to_owned)
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(cited, BTreeSet::from([restored]));
}

#[test]
fn b2d_owner_restatement_that_differs_across_lineages_refuses() {
    let (publication, derived) = b2d_failed_prefix();
    let second = captured("capture:second", derived);
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(
        &publication,
        vec![Step::Exhausted, Step::Exhausted, Step::Propose(3)],
    );
    source.seed_shift = 1;
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let abort = report(
        run_serial(
            &second,
            &mut source,
            None,
            Box::new(&mut buffer),
            Some((&resume, &mut progress)),
        )
        .unwrap_err(),
    );
    assert!(abort.primary_error.message.contains("seed_conflict"));
    assert_eq!(
        abort.failed_attempt.failure_stage,
        Some(FailureStage::ProposalConversion)
    );
    assert_eq!(source.transitions, vec![owner_ref(2)], "the ask is cleaned");
    assert_no_completion(&buffer);
}

#[test]
fn b2d_same_seed_twice_in_one_proposal_refuses_before_the_stream_records_it() {
    let request = budget("capture:first", 1);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    source.duplicate_resolution = true;
    let mut buffer = Vec::new();
    let abort =
        report(run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap_err());
    assert!(abort
        .primary_error
        .message
        .contains("seed_resolution_conflict"));
    assert_eq!(
        abort.failed_attempt.failure_stage,
        Some(FailureStage::ProposalConversion)
    );
    assert_eq!(source.transitions, vec![owner_ref(0)]);
    assert_no_completion(&buffer);
    assert!(
        stream_events(&buffer)
            .iter()
            .all(|event| event["kind"] != "effective_seed_resolved"),
        "neither value may be recorded as the seed's fact"
    );
}

#[test]
fn b2d_abort_names_retained_running_trials_without_touching_them() {
    let publication = b2c_regression_pending_publication(3);
    let second = captured("capture:second", Vec::new());
    let resume = options(Some(publication.clone()));
    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = Progress::default();
    let mut buffer = Vec::new();
    let abort = report(
        run_serial(
            &second,
            &mut source,
            Some(TRIAL_1),
            Box::new(&mut buffer),
            Some((&resume, &mut progress)),
        )
        .unwrap_err(),
    );
    assert_eq!(abort.uncaptured_pending, vec![2]);
    assert_eq!(
        source.transitions,
        vec![owner_ref(1)],
        "only trial 1 failed"
    );
    assert_eq!(source.fails.len(), 1, "trial 2 got no cleanup call");
    assert_eq!(source.store[&owner_ref(2)], TrialStatus::Running);
    assert_no_completion(&buffer);
    let events = stream_events(&buffer);
    assert!(
        events.iter().all(|event| event["kind"] != "search_close"),
        "a retained RUNNING trial keeps the Search open"
    );
    assert!(
        events.iter().any(|event| {
            event["kind"] == "trial_resume" && event["scope_id"] == "proposal:0000000002"
        }),
        "the prefix shows the trial it could not close"
    );
}

#[test]
fn b2d_non_durable_session_refuses_worker_windows_before_any_ask() {
    let request = captured("capture:first", Vec::new());
    let plan = plan();
    let resume = options(None);
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Propose(2)]);
    let mut buffer = Vec::new();
    let session =
        HostHpoCaptureSession::open(&plan, &request, None, &mut source, Box::new(&mut buffer))
            .unwrap();
    assert!(session
        .prepare_worker_window(&plan, &request, &resume, &mut source, 2)
        .is_err());
    assert!(source.asks.is_empty());
}

#[test]
fn b2d_final_publication_pairs_the_state_exhaustion_left() {
    let request = captured("capture:first", Vec::new());
    let mut source = Source::scripted(vec![Step::Propose(1), Step::Exhausted]);
    source.exhaustion_moves_state = true;
    let mut progress = Progress::default();
    let resume = options(None);
    let mut buffer = Vec::new();
    let outcome = run_serial(
        &request,
        &mut source,
        None,
        Box::new(&mut buffer),
        Some((&resume, &mut progress)),
    )
    .unwrap();
    assert_eq!(outcome.outcome.status, HostHpoSearchStatus::Exhausted);
    verified_counts(&buffer, &outcome);
    let publication = outcome.publication.unwrap();
    let digest = source.state_digest();
    assert_eq!(
        publication.owner_checkpoint_fingerprint.as_deref(),
        Some(digest.as_str())
    );
    let observed = source.proposal_state().unwrap();
    validate_host_hpo_capture_lineage(
        &captured("capture:second", Vec::new()),
        Some(&publication),
        Some(&observed),
    )
    .unwrap();
}

#[test]
fn b2d_uncaptured_outcome_reports_how_the_stream_ended() {
    let publication = b2c_regression_pending_publication(3);
    let second = captured("capture:second", Vec::new());
    let resume = options(Some(publication.clone()));
    let cancelled = || Progress {
        cancel_after: Some(1),
        ..Progress::default()
    };
    let factory = ControllerFactory::new(None);

    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = cancelled();
    let mut buffer = Vec::new();
    let outcome = run_parallel(
        &second,
        &mut source,
        &factory,
        Box::new(&mut buffer),
        3,
        Some((&resume, &mut progress)),
    )
    .unwrap();
    let refusal = outcome.refusal.unwrap();
    assert_eq!(refusal.refusal_code.as_deref(), Some("empty_or_unclosed"));
    assert_eq!(refusal.flush, HostHpoCaptureFlush::Flushed);
    assert!(!buffer.is_empty(), "the prefix reached the sink");

    let mut source = Source::at_publication(&publication, Vec::new());
    let mut progress = cancelled();
    let outcome = run_parallel(
        &second,
        &mut source,
        &factory,
        Box::new(BrokenSink),
        3,
        Some((&resume, &mut progress)),
    )
    .unwrap();
    let refusal = outcome.refusal.unwrap();
    assert_eq!(refusal.flush, HostHpoCaptureFlush::Failed);
    assert!(!outcome.evidence_errors.is_empty(), "the host can see it");
    assert!(outcome.completion.is_none());
}

#[test]
fn b2d_completed_stream_reports_no_refusal() {
    let request = budget("capture:first", 1);
    let mut source = Source::scripted(vec![Step::Propose(1)]);
    let mut buffer = Vec::new();
    let outcome = run_serial(&request, &mut source, None, Box::new(&mut buffer), None).unwrap();
    assert!(outcome.refusal.is_none());
    assert!(outcome.uncaptured_pending.is_empty());
    verified_counts(&buffer, &outcome);
}
