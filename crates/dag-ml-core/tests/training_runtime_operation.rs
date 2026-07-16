use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use dag_ml_core::training::PredictionSource;
use dag_ml_core::*;
use sha2::{Digest, Sha256};

const REQUEST_FIXTURE: &str =
    include_str!("../../../examples/fixtures/training/training_request_refit.v1.json");
const PACKAGE_FIXTURE: &str =
    include_str!("../../../examples/fixtures/training/portable_predictor_package.v1.json");

#[derive(Default)]
struct CallState {
    calls: Mutex<Vec<(Phase, NodeId)>>,
    fit_counts: Mutex<BTreeMap<VariantId, usize>>,
    next_handle: AtomicU64,
    preferred: Mutex<Option<VariantId>>,
    divergent_rerun: Mutex<bool>,
    invalid_refit_output: Mutex<bool>,
    score_auxiliary: Mutex<bool>,
    emit_extra_fit_cv_partitions: Mutex<bool>,
    emit_explicit_model_ports: Mutex<bool>,
    observed_model_patch_values: Mutex<Vec<Option<serde_json::Value>>>,
}

impl CallState {
    fn count(&self, phase: Phase, node: &str) -> usize {
        self.calls
            .lock()
            .unwrap()
            .iter()
            .filter(|(actual_phase, actual_node)| {
                *actual_phase == phase && actual_node.as_str() == node
            })
            .count()
    }

    fn total(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    fn handle(&self) -> u64 {
        self.next_handle.fetch_add(1, Ordering::SeqCst) + 1
    }
}

struct AttestedProvider {
    identity: TrainingDataIdentity,
    relations: SampleRelationSet,
    contradictory_relations: Option<SampleRelationSet>,
    omit_relations: bool,
    next_handle: AtomicU64,
}

impl RuntimeDataProvider for AttestedProvider {
    fn materialize(&self, _request: &DataMaterializationRequest) -> Result<HandleRef> {
        Ok(self.handle(HandleKind::Data))
    }

    fn make_view(&self, _request: &DataViewRequest) -> Result<HandleRef> {
        Ok(self.handle(HandleKind::DataView))
    }

    fn training_data_identity(
        &self,
        _binding: &DataBinding,
    ) -> Result<Option<TrainingDataIdentity>> {
        Ok(Some(self.identity.clone()))
    }

    fn coordinator_relations(&self, _binding: &DataBinding) -> Result<Option<SampleRelationSet>> {
        if self.omit_relations {
            return Ok(None);
        }
        Ok(Some(
            self.contradictory_relations
                .clone()
                .unwrap_or_else(|| self.relations.clone()),
        ))
    }
}

impl AttestedProvider {
    fn handle(&self, kind: HandleKind) -> HandleRef {
        HandleRef {
            handle: self.next_handle.fetch_add(1, Ordering::SeqCst) + 1,
            kind,
            owner_controller: ControllerId::new("controller:data.provider").unwrap(),
        }
    }
}

struct TrainingController {
    id: ControllerId,
    state: Arc<CallState>,
    emits_predictions: bool,
    emits_artifact: bool,
    prediction_name: String,
}

impl RuntimeController for TrainingController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        self.state
            .calls
            .lock()
            .unwrap()
            .push((task.phase, task.node_plan.node_id.clone()));
        let is_model = task.node_plan.node_id.as_str() == "model:base";
        if is_model {
            self.state
                .observed_model_patch_values
                .lock()
                .unwrap()
                .push(task.node_plan.params.get("patched_bias").cloned());
        }
        let sample_ids = match task.fold_id.as_ref().map(FoldId::as_str) {
            Some("fold:0") => vec![sample("sample:1"), sample("sample:2")],
            Some("fold:1") => vec![sample("sample:3"), sample("sample:4")],
            _ => (1..=4)
                .map(|index| sample(&format!("sample:{index}")))
                .collect(),
        };

        let mut value = 0.0;
        if is_model && task.phase == Phase::FitCv {
            let variant = task.variant_id.clone().expect("campaign task variant");
            let preferred = self.state.preferred.lock().unwrap().clone().unwrap();
            let ordinal = {
                let mut counts = self.state.fit_counts.lock().unwrap();
                let count = counts.entry(variant.clone()).or_default();
                let ordinal = *count;
                *count += 1;
                ordinal
            };
            value = if variant == preferred { 0.0 } else { 5.0 };
            if variant == preferred && ordinal >= 2 && *self.state.divergent_rerun.lock().unwrap() {
                value = 1.0;
            }
        } else if self.emits_predictions
            && task.phase == Phase::FitCv
            && *self.state.score_auxiliary.lock().unwrap()
        {
            let preferred = self.state.preferred.lock().unwrap().clone().unwrap();
            value = if task.variant_id.as_ref() == Some(&preferred) {
                5.0
            } else {
                0.0
            };
        }

        let partition = if matches!(task.phase, Phase::Refit | Phase::Predict) {
            PredictionPartition::Final
        } else {
            PredictionPartition::Validation
        };
        let prediction_target = if task.phase == Phase::Refit
            && *self.state.invalid_refit_output.lock().unwrap()
            && is_model
        {
            "wrong"
        } else {
            &self.prediction_name
        };
        let explicit_model_ports =
            is_model && *self.state.emit_explicit_model_ports.lock().unwrap();
        let mut predictions = if self.emits_predictions
            && matches!(task.phase, Phase::FitCv | Phase::Refit | Phase::Predict)
        {
            vec![PredictionBlock {
                prediction_id: Some(format!(
                    "prediction:{}:{}:{}",
                    task.node_plan.node_id,
                    task.phase.as_str(),
                    task.fold_id.as_ref().map(FoldId::as_str).unwrap_or("full")
                )),
                producer_node: task.node_plan.node_id.clone(),
                producer_port: explicit_model_ports.then(|| "oof".to_string()),
                partition,
                fold_id: if task.phase == Phase::FitCv {
                    task.fold_id.clone()
                } else {
                    None
                },
                sample_ids: sample_ids.clone(),
                values: sample_ids.iter().map(|_| vec![value]).collect(),
                target_names: vec![prediction_target.to_string()],
            }]
        } else {
            Vec::new()
        };
        if explicit_model_ports && !predictions.is_empty() {
            let mut sibling = predictions
                .first()
                .expect("explicit model port controller emits the primary prediction")
                .clone();
            sibling.prediction_id = sibling
                .prediction_id
                .as_ref()
                .map(|id| format!("{id}:probability"));
            sibling.producer_port = Some("probability".to_string());
            sibling.values = sibling
                .values
                .iter()
                .map(|row| row.iter().map(|value| value + 100.0).collect())
                .collect();
            predictions.push(sibling);
        }
        if self.emits_predictions
            && task.phase == Phase::FitCv
            && *self.state.emit_extra_fit_cv_partitions.lock().unwrap()
        {
            let validation = predictions
                .first()
                .expect("FIT_CV prediction controller emits Validation")
                .clone();
            predictions.extend([
                PredictionBlock {
                    prediction_id: Some(format!(
                        "prediction:{}:FIT_CV:train:{}",
                        task.node_plan.node_id,
                        task.fold_id.as_ref().map(FoldId::as_str).unwrap_or("full")
                    )),
                    partition: PredictionPartition::Train,
                    ..validation.clone()
                },
                PredictionBlock {
                    prediction_id: Some(format!(
                        "prediction:{}:FIT_CV:test:{}",
                        task.node_plan.node_id,
                        task.fold_id.as_ref().map(FoldId::as_str).unwrap_or("full")
                    )),
                    partition: PredictionPartition::Test,
                    ..validation.clone()
                },
                PredictionBlock {
                    prediction_id: Some(format!(
                        "prediction:{}:FIT_CV:final:{}",
                        task.node_plan.node_id,
                        task.fold_id.as_ref().map(FoldId::as_str).unwrap_or("full")
                    )),
                    partition: PredictionPartition::Final,
                    ..validation
                },
            ]);
        }
        let explanations = if is_model && task.phase == Phase::Explain {
            vec![ExplanationBlock {
                producer_node: task.node_plan.node_id.clone(),
                producer_port: explicit_model_ports.then(|| "oof".to_string()),
                method: "fixture_explain".to_string(),
                target_name: Some(self.prediction_name.clone()),
                payload: serde_json::json!({"importance": 1.0}),
            }]
        } else {
            Vec::new()
        };
        let score_this_producer =
            is_model || (self.emits_predictions && *self.state.score_auxiliary.lock().unwrap());
        let regression_targets = if score_this_producer && task.phase == Phase::FitCv {
            vec![RegressionTargetBlock {
                level: PredictionLevel::Sample,
                unit_ids: sample_ids
                    .iter()
                    .cloned()
                    .map(PredictionUnitId::Sample)
                    .collect(),
                values: sample_ids.iter().map(|_| vec![0.0]).collect(),
                target_names: vec![self.prediction_name.clone()],
            }]
        } else {
            Vec::new()
        };
        let artifacts = if self.emits_artifact && task.phase == Phase::Refit {
            vec![ArtifactRef {
                id: ArtifactId::new(format!("artifact:{}:refit", task.node_plan.node_id)).unwrap(),
                kind: "test_model".to_string(),
                controller_id: self.id.clone(),
                backend: None,
                uri: None,
                content_fingerprint: None,
                size_bytes: Some(8),
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
                        handle: self.state.handle(),
                        kind: HandleKind::Artifact,
                        owner_controller: self.id.clone(),
                    },
                )
            })
            .collect();
        let output_port = if is_model { "oof" } else { "x_out" };
        Ok(NodeResult {
            schema_version: None,
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                output_port.to_string(),
                HandleRef {
                    handle: self.state.handle(),
                    kind: if is_model {
                        HandleKind::Prediction
                    } else {
                        HandleKind::Data
                    },
                    owner_controller: self.id.clone(),
                },
            )]),
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            explanations,
            shape_deltas: Vec::new(),
            artifacts: artifacts.clone(),
            artifact_handles,
            fit_influence_diagnostics: Vec::new(),
            regression_targets,
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:{}:{}:{}:{}",
                    task.node_plan.node_id,
                    task.phase.as_str(),
                    task.variant_id
                        .as_ref()
                        .map(VariantId::as_str)
                        .unwrap_or("base"),
                    task.fold_id.as_ref().map(FoldId::as_str).unwrap_or("full")
                ))
                .unwrap(),
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
                loss_attestations: Vec::new(),
            },
        })
    }
}

struct Fixture {
    request: TrainingRequest,
    relations: SampleRelationSet,
    influence: TrainingInfluenceManifest,
    preferred: VariantId,
}

fn fixture(refit: bool, stacking: bool) -> Fixture {
    let mut request_json: serde_json::Value = serde_json::from_str(REQUEST_FIXTURE).unwrap();
    request_json["options"]["selection_output_id"] =
        serde_json::Value::String("output:prediction".to_string());
    let mut request: TrainingRequest = serde_json::from_value(request_json).unwrap();
    request.options.refit = refit;
    request.options.refit_strategy = refit.then_some(RefitStrategy::RefitOne);
    request.options.selection.required_metric_level = Some(PredictionLevel::Sample);
    request.options.selection.evaluation_scope = Some(EvaluationScope::Oof);
    request.options.resources.memory_bytes = None;
    request.options.resources.wall_time_ms = None;
    request.options.artifacts.cv_artifacts = CvArtifactRetention::Discard;
    request.options.artifacts.fitted_artifacts = FittedArtifactMode::AllowHostSidecar;
    for dimension in &mut request.campaign.generation.dimensions {
        if dimension.name == "model_family" {
            for (index, choice) in dimension.choices.iter_mut().enumerate() {
                choice.param_overrides = vec![GenerationParamOverride {
                    node_id: node("model:base"),
                    params: BTreeMap::from([(
                        "n_estimators".to_string(),
                        serde_json::json!(10 + index),
                    )]),
                }];
            }
        }
    }
    if stacking {
        add_stacking_edge(&mut request);
    }
    let relations = relations();
    let relation_fingerprint = relations.fingerprint().unwrap();
    for bindings in request.campaign.data_bindings.values_mut() {
        for binding in bindings {
            binding.relation_fingerprint = Some(relation_fingerprint.clone());
        }
    }
    for identity in &mut request.data_identities {
        identity.relation_fingerprint = relation_fingerprint.clone();
        identity.identity_fingerprint = "0".repeat(64);
        identity.identity_fingerprint = identity.compute_fingerprint().unwrap();
    }
    resign_request(&mut request);
    let projection = request.project().unwrap();
    let preferred = projection.plan.variants[0].variant_id.clone();
    let influence = influence_manifest(&request, &projection, &relations);
    Fixture {
        request,
        relations,
        influence,
        preferred,
    }
}

fn add_stacking_edge(request: &mut TrainingRequest) {
    let prediction_port = PortSpec {
        name: "oof_aux".to_string(),
        kind: PortKind::Prediction,
        representation: None,
        cardinality: PortCardinality::One,
        unit_level: None,
        alignment_key: None,
        target_level: None,
        description: String::new(),
    };
    let input_port = PortSpec {
        name: "meta".to_string(),
        ..prediction_port.clone()
    };
    request
        .graph
        .nodes
        .iter_mut()
        .find(|node| node.id.as_str() == "transform:snv")
        .unwrap()
        .ports
        .outputs
        .push(prediction_port.clone());
    request
        .graph
        .nodes
        .iter_mut()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .inputs
        .push(input_port.clone());
    request.graph.edges.push(EdgeSpec {
        source: PortRef {
            node_id: node("transform:snv"),
            port_name: "oof_aux".to_string(),
        },
        target: PortRef {
            node_id: node("model:base"),
            port_name: "meta".to_string(),
        },
        contract: EdgeContract {
            requires_oof: true,
            requires_fold_alignment: true,
            ..EdgeContract::new(PortKind::Prediction, None)
        },
    });
    let transform = request
        .controller_manifests
        .iter_mut()
        .find(|manifest| manifest.controller_id.as_str() == "controller:transform.mock")
        .unwrap();
    transform.output_ports.push(prediction_port);
    transform
        .capabilities
        .insert(ControllerCapability::EmitsPredictions);
    let model = request
        .controller_manifests
        .iter_mut()
        .find(|manifest| manifest.controller_id.as_str() == "controller:model.mock")
        .unwrap();
    model.input_ports.push(input_port);
}

fn run(
    fixture: &Fixture,
    state: Arc<CallState>,
    provider: &AttestedProvider,
    store: &mut InMemoryArtifactStore,
) -> Result<TrainingOutcome> {
    run_custom(
        fixture,
        state,
        provider,
        store,
        "outcome:test.native",
        BTreeMap::from([("test".to_string(), serde_json::json!(true))]),
        true,
    )
}

fn run_custom(
    fixture: &Fixture,
    state: Arc<CallState>,
    provider: &AttestedProvider,
    store: &mut InMemoryArtifactStore,
    outcome_id: &str,
    diagnostics: BTreeMap<String, serde_json::Value>,
    complete_controllers: bool,
) -> Result<TrainingOutcome> {
    *state.preferred.lock().unwrap() = Some(fixture.preferred.clone());
    let controllers = controllers(fixture, state, complete_controllers);
    execute_training(TrainingExecutionInput {
        request: &fixture.request,
        outcome_id: outcome_id.to_string(),
        run_id: RunId::new("run:test.native").unwrap(),
        bundle_id: BundleId::new("bundle:test.native").unwrap(),
        controllers: &controllers,
        data_provider: provider,
        relations: &fixture.relations,
        training_influence: &fixture.influence,
        artifact_store: store,
        warnings: Vec::new(),
        diagnostics,
    })
}

fn controllers(
    fixture: &Fixture,
    state: Arc<CallState>,
    complete: bool,
) -> RuntimeControllerRegistry {
    let transform_predictions = fixture
        .request
        .graph
        .edges
        .iter()
        .any(|edge| edge.contract.requires_oof);
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(TrainingController {
            id: ControllerId::new("controller:transform.mock").unwrap(),
            state: state.clone(),
            emits_predictions: transform_predictions,
            emits_artifact: false,
            prediction_name: "aux".to_string(),
        }))
        .unwrap();
    if complete {
        controllers
            .register(Box::new(TrainingController {
                id: ControllerId::new("controller:model.mock").unwrap(),
                state,
                emits_predictions: true,
                emits_artifact: true,
                prediction_name: "protein".to_string(),
            }))
            .unwrap();
    }
    controllers
}

fn provider(fixture: &Fixture) -> AttestedProvider {
    AttestedProvider {
        identity: fixture.request.data_identities[0].clone(),
        relations: fixture.relations.clone(),
        contradictory_relations: None,
        omit_relations: false,
        next_handle: AtomicU64::new(0),
    }
}

fn replay_envelopes_with_relation(
    outcome: &TrainingOutcome,
    relation_fingerprint: &str,
) -> BTreeMap<String, ExternalDataPlanEnvelope> {
    outcome
        .execution_bundle
        .data_requirements
        .iter()
        .map(|requirement| {
            let key = requirement.key();
            (
                key.clone(),
                ExternalDataPlanEnvelope {
                    schema_version: EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION,
                    schema_fingerprint: requirement.schema_fingerprint.clone(),
                    plan_fingerprint: requirement.plan_fingerprint.clone(),
                    relation_fingerprint: Some(relation_fingerprint.to_string()),
                    data_content_fingerprint: Some(content_hash(&format!("{key}:data"))),
                    target_content_fingerprint: Some(content_hash(&format!("{key}:target"))),
                    coordinator_relations: Some(relations()),
                },
            )
        })
        .collect()
}

fn replay_request(outcome: &TrainingOutcome, phase: Phase) -> TrainingReplayRequest {
    let mut data_envelope_keys = outcome
        .execution_bundle
        .data_requirements
        .iter()
        .map(|requirement| requirement.key())
        .collect::<Vec<_>>();
    data_envelope_keys.sort();
    let mut output_binding_ids = outcome
        .outputs
        .iter()
        .map(|output| output.binding.binding_id.clone())
        .collect::<Vec<_>>();
    output_binding_ids.sort();
    let mut request = TrainingReplayRequest {
        schema_version: TRAINING_REPLAY_REQUEST_SCHEMA_VERSION,
        request_id: format!("replay:attached.{}", phase.as_str().to_ascii_lowercase()),
        source_outcome_fingerprint: outcome.outcome_fingerprint.clone(),
        phase,
        data_envelope_keys,
        output_binding_ids,
        request_fingerprint: "0".repeat(64),
    };
    request.request_fingerprint = request.compute_fingerprint().unwrap();
    request
}

fn add_model_probability_port(fixture: &mut Fixture) {
    let mut extra = fixture
        .request
        .graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .outputs[0]
        .clone();
    extra.name = "probability".to_string();
    fixture
        .request
        .graph
        .nodes
        .iter_mut()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .outputs
        .push(extra.clone());
    fixture
        .request
        .controller_manifests
        .iter_mut()
        .find(|manifest| manifest.controller_id.as_str() == "controller:model.mock")
        .unwrap()
        .output_ports
        .push(extra);
    fixture.request.options.outputs[0].port_name = Some("oof".to_string());
    rebuild(fixture);
}

fn add_explain_support(fixture: &mut Fixture) {
    for manifest in &mut fixture.request.controller_manifests {
        manifest.supported_phases.insert(Phase::Explain);
    }
    rebuild(fixture);
}

fn content_hash(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn assert_preflight_rejected(mut fixture: Fixture, mutate: impl FnOnce(&mut TrainingRequest)) {
    mutate(&mut fixture.request);
    resign_request(&mut fixture.request);
    let state = Arc::new(CallState::default());
    let mut store = InMemoryArtifactStore::new();
    assert!(run(&fixture, state.clone(), &provider(&fixture), &mut store).is_err());
    assert_eq!(state.total(), 0);
    assert!(store.is_empty());
}

fn early_stopping_requirements() -> Vec<ControllerInfluenceRequirement> {
    vec![
        ControllerInfluenceRequirement {
            node_id: node("model:base"),
            kind: TrainingInfluenceKind::EarlyStopping,
            scope_id: "early:fold:0".to_string(),
            phase: Phase::FitCv,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            physical_sample_ids: vec![sample("sample:3")],
        },
        ControllerInfluenceRequirement {
            node_id: node("model:base"),
            kind: TrainingInfluenceKind::EarlyStopping,
            scope_id: "early:fold:1".to_string(),
            phase: Phase::FitCv,
            fold_id: Some(FoldId::new("fold:1").unwrap()),
            physical_sample_ids: vec![sample("sample:1")],
        },
        ControllerInfluenceRequirement {
            node_id: node("model:base"),
            kind: TrainingInfluenceKind::EarlyStopping,
            scope_id: "early:refit".to_string(),
            phase: Phase::Refit,
            fold_id: None,
            physical_sample_ids: vec![sample("sample:1")],
        },
    ]
}

fn full_scope_requirements(
    kind: TrainingInfluenceKind,
    prefix: &str,
) -> Vec<ControllerInfluenceRequirement> {
    vec![
        ControllerInfluenceRequirement {
            node_id: node("model:base"),
            kind,
            scope_id: format!("{prefix}:fold:0"),
            phase: Phase::FitCv,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            physical_sample_ids: vec![sample("sample:3"), sample("sample:4")],
        },
        ControllerInfluenceRequirement {
            node_id: node("model:base"),
            kind,
            scope_id: format!("{prefix}:fold:1"),
            phase: Phase::FitCv,
            fold_id: Some(FoldId::new("fold:1").unwrap()),
            physical_sample_ids: vec![sample("sample:1"), sample("sample:2")],
        },
        ControllerInfluenceRequirement {
            node_id: node("model:base"),
            kind,
            scope_id: format!("{prefix}:refit"),
            phase: Phase::Refit,
            fold_id: None,
            physical_sample_ids: (1..=4)
                .map(|index| sample(&format!("sample:{index}")))
                .collect(),
        },
    ]
}

fn relations() -> SampleRelationSet {
    SampleRelationSet {
        records: (1..=4)
            .map(|index| {
                let mut relation = SampleRelation::new(
                    ObservationId::new(format!("observation:{index}")).unwrap(),
                    sample(&format!("sample:{index}")),
                );
                relation.group_id =
                    Some(GroupId::new(if index <= 2 { "group:0" } else { "group:1" }).unwrap());
                relation
            })
            .collect(),
    }
}

fn influence_manifest(
    request: &TrainingRequest,
    projection: &TrainingContractProjection,
    relations: &SampleRelationSet,
) -> TrainingInfluenceManifest {
    TrainingInfluenceManifest::derive_for_projection(projection, request, relations).unwrap()
}

fn resign_request(request: &mut TrainingRequest) {
    request.request_fingerprint = "0".repeat(64);
    request.request_fingerprint = request.compute_fingerprint().unwrap();
}

fn rebuild(fixture: &mut Fixture) {
    resign_request(&mut fixture.request);
    let projection = fixture.request.project().unwrap();
    fixture.preferred = projection.plan.variants[0].variant_id.clone();
    fixture.influence = influence_manifest(&fixture.request, &projection, &fixture.relations);
}

fn resign_outcome(outcome: &mut TrainingOutcome) {
    let plan_json = serde_json::to_string(&outcome.effective_plan).unwrap();
    outcome.effective_plan_fingerprint =
        parse_typed_json(&plan_json).unwrap().fingerprint().unwrap();
    outcome.outcome_fingerprint = "0".repeat(64);
    outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
}

fn legacy_serde_fingerprint(value: &impl serde::Serialize) -> String {
    let digest = Sha256::digest(serde_json::to_vec(value).unwrap());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn typed_fingerprint(value: &impl serde::Serialize) -> String {
    let json = serde_json::to_string(value).unwrap();
    parse_typed_json(&json).unwrap().fingerprint().unwrap()
}

fn sample(value: &str) -> SampleId {
    SampleId::new(value).unwrap()
}

fn node(value: &str) -> NodeId {
    NodeId::new(value).unwrap()
}

#[test]
fn native_training_refit_and_no_refit_are_deterministic_and_auditable() {
    let refit = fixture(true, false);
    let state = Arc::new(CallState::default());
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(&refit, state.clone(), &provider(&refit), &mut store).unwrap();
    outcome.validate().unwrap();
    let reference = outcome.to_reference().unwrap();
    assert_eq!(
        reference.training_request_fingerprint,
        refit.request.request_fingerprint
    );
    assert_eq!(
        reference.data_identities_fingerprint,
        outcome.data_identities_fingerprint().unwrap()
    );
    assert_eq!(
        reference.execution_bundle_fingerprint,
        outcome.execution_bundle_fingerprint().unwrap()
    );
    assert_eq!(outcome.refit.status, TrainingRefitStatus::Completed);
    // A completed refit whose closure supports PREDICT (but not EXPLAIN) and whose
    // only state-retaining node (model:base) has its retained artifact advertises
    // exactly [PREDICT] and never re-advertises REFIT.
    assert_eq!(outcome.replayable_phases, vec![Phase::Predict]);
    assert_eq!(
        outcome.outputs[0].binding.prediction_source,
        PredictionSource::FinalRefit
    );
    assert_eq!(outcome.execution_bundle.selections.len(), 1);
    assert_eq!(outcome.execution_bundle.refit_artifacts.len(), 1);
    assert_eq!(store.len(), 1);
    assert_eq!(state.count(Phase::FitCv, "model:base"), 6);
    assert_eq!(state.count(Phase::FitCv, "transform:snv"), 6);
    assert_eq!(state.count(Phase::Refit, "model:base"), 1);
    assert_eq!(state.count(Phase::Refit, "transform:snv"), 1);
    assert_eq!(outcome.parameter_patches.len(), 1);
    assert_eq!(
        outcome.effective_plan.node_plans[&node("model:base")].params["n_estimators"],
        outcome.parameter_patches[0].value
    );

    let no_refit = fixture(false, false);
    let no_refit_state = Arc::new(CallState::default());
    let mut no_refit_store = InMemoryArtifactStore::new();
    let first = run(
        &no_refit,
        no_refit_state.clone(),
        &provider(&no_refit),
        &mut no_refit_store,
    )
    .unwrap();
    let mut second_store = InMemoryArtifactStore::new();
    let second = run(
        &no_refit,
        Arc::new(CallState::default()),
        &provider(&no_refit),
        &mut second_store,
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.refit.status, TrainingRefitStatus::Skipped);
    assert_eq!(
        first.outputs[0].binding.prediction_source,
        PredictionSource::CvEnsemble
    );
    assert_eq!(first.replayable_phases, vec![Phase::Refit]);
    assert!(no_refit_store.is_empty());
    assert_eq!(no_refit_state.count(Phase::Refit, "model:base"), 0);
}

#[test]
fn attached_training_replay_predict_rebinds_current_cohort_without_mutating_source() {
    let mut fixture = fixture(true, false);
    add_model_probability_port(&mut fixture);
    let state = Arc::new(CallState::default());
    *state.emit_explicit_model_ports.lock().unwrap() = true;
    let mut store = InMemoryArtifactStore::new();
    let source = run(&fixture, state.clone(), &provider(&fixture), &mut store)
        .expect("source training outcome");
    assert!(source.replayable_phases.contains(&Phase::Predict));
    let source_fingerprint = source.outcome_fingerprint.clone();
    let source_bundle_relation = source.execution_bundle.data_requirements[0]
        .relation_fingerprint
        .clone();

    let current_relation = "f".repeat(64);
    let envelopes = replay_envelopes_with_relation(&source, &current_relation);
    let request = replay_request(&source, Phase::Predict);
    let controllers = controllers(&fixture, state.clone(), true);

    let replay = execute_attached_training_replay(AttachedTrainingReplayInput {
        source: &source,
        request: &request,
        outcome_id: "replay:attached.predict.outcome".to_string(),
        run_id: RunId::new("run:attached.predict").unwrap(),
        controllers: &controllers,
        data_provider: &provider(&fixture),
        artifact_store: &store,
        data_envelopes: &envelopes,
        warnings: Vec::new(),
        diagnostics: BTreeMap::from([("attached".to_string(), serde_json::json!(true))]),
    })
    .expect("attached replay");

    assert_eq!(source.outcome_fingerprint, source_fingerprint);
    assert_eq!(
        source.execution_bundle.data_requirements[0].relation_fingerprint,
        source_bundle_relation
    );
    assert_eq!(replay.phase, Phase::Predict);
    assert_eq!(
        replay.source_training_outcome,
        source.to_reference().unwrap()
    );
    assert_eq!(
        replay.replay_request_fingerprint,
        request.request_fingerprint
    );
    assert_eq!(replay.outputs.len(), request.output_binding_ids.len());
    assert!(replay.explanations.is_empty());
    assert!(replay
        .input_data_identities
        .iter()
        .all(|identity| identity.relation_fingerprint == current_relation));
    assert!(replay.outputs.iter().all(|output| {
        output.schema_version == Some(BOUND_TRAINING_OUTPUT_SCHEMA_VERSION)
            && !output.predictions.is_empty()
            && output.predictions.iter().all(|block| {
                block.partition == PredictionPartition::Final
                    && block.fold_id.is_none()
                    && block.producer_port.as_deref() == Some(output.binding.port_name.as_str())
            })
    }));
    assert_eq!(
        replay.prediction_block_count,
        replay
            .outputs
            .iter()
            .map(|output| output.predictions.len())
            .sum::<usize>()
    );
    assert!(state.count(Phase::Predict, "model:base") > 0);
}

#[test]
fn attached_training_replay_explain_emits_explanations_without_outputs() {
    let mut fixture = fixture(true, false);
    add_model_probability_port(&mut fixture);
    add_explain_support(&mut fixture);
    let state = Arc::new(CallState::default());
    *state.emit_explicit_model_ports.lock().unwrap() = true;
    let mut store = InMemoryArtifactStore::new();
    let source = run(&fixture, state.clone(), &provider(&fixture), &mut store)
        .expect("source training outcome");
    assert!(source.replayable_phases.contains(&Phase::Explain));

    let current_relation = "e".repeat(64);
    let envelopes = replay_envelopes_with_relation(&source, &current_relation);
    let request = replay_request(&source, Phase::Explain);
    let controllers = controllers(&fixture, state.clone(), true);

    let replay = execute_attached_training_replay(AttachedTrainingReplayInput {
        source: &source,
        request: &request,
        outcome_id: "replay:attached.explain.outcome".to_string(),
        run_id: RunId::new("run:attached.explain").unwrap(),
        controllers: &controllers,
        data_provider: &provider(&fixture),
        artifact_store: &store,
        data_envelopes: &envelopes,
        warnings: Vec::new(),
        diagnostics: BTreeMap::new(),
    })
    .expect("attached explain replay");

    assert_eq!(replay.phase, Phase::Explain);
    assert!(replay.outputs.is_empty());
    assert_eq!(replay.explanation_block_count, replay.explanations.len());
    assert!(replay.explanations.iter().any(|block| {
        block.producer_node.as_str() == "model:base"
            && block.producer_port.as_deref() == Some("oof")
            && block.method == "fixture_explain"
    }));
    assert!(replay
        .input_data_identities
        .iter()
        .all(|identity| identity.relation_fingerprint == current_relation));
    assert!(state.count(Phase::Explain, "model:base") > 0);
}

#[test]
fn cv_ensemble_excludes_non_validation_fit_cv_blocks_and_refit_stays_final() {
    let no_refit = fixture(false, false);
    let no_refit_state = Arc::new(CallState::default());
    *no_refit_state.emit_extra_fit_cv_partitions.lock().unwrap() = true;
    let mut no_refit_store = InMemoryArtifactStore::new();
    let no_refit_outcome = run(
        &no_refit,
        no_refit_state.clone(),
        &provider(&no_refit),
        &mut no_refit_store,
    )
    .unwrap();
    let cv_output = &no_refit_outcome.outputs[0];
    assert_eq!(
        cv_output.binding.prediction_source,
        PredictionSource::CvEnsemble
    );
    assert_eq!(
        cv_output.predictions.len(),
        2,
        "the selected rerun keeps one Validation block per fold"
    );
    assert!(
        cv_output
            .predictions
            .iter()
            .all(|block| block.partition == PredictionPartition::Validation),
        "Train/Test/Final FIT_CV blocks must not enter a cv_ensemble output"
    );
    assert!(
        !cv_output.aggregated_predictions.is_empty()
            && cv_output
                .aggregated_predictions
                .iter()
                .all(|block| block.partition == PredictionPartition::Validation),
        "the Validation OOF average remains present"
    );
    assert!(no_refit_state.count(Phase::FitCv, "model:base") > 0);

    for invalid_partition in [
        PredictionPartition::Train,
        PredictionPartition::Test,
        PredictionPartition::Final,
    ] {
        let mut tampered = no_refit_outcome.clone();
        tampered.outputs[0].predictions[0].partition = invalid_partition.clone();
        resign_outcome(&mut tampered);
        let error = TrainingOutcome::from_json(&serde_json::to_string(&tampered).unwrap())
            .expect_err("a re-signed cv_ensemble cannot contain a non-Validation block");
        assert!(
            error
                .to_string()
                .contains("cv_ensemble output blocks must use validation partition"),
            "unexpected {invalid_partition:?} rejection: {error}"
        );
    }
    let mut missing_fold = no_refit_outcome.clone();
    missing_fold.outputs[0].predictions[0].fold_id = None;
    resign_outcome(&mut missing_fold);
    let error = TrainingOutcome::from_json(&serde_json::to_string(&missing_fold).unwrap())
        .expect_err("a cv_ensemble Validation block must identify its fold or avg reduction");
    assert!(
        error
            .to_string()
            .contains("cv_ensemble output blocks must use validation partition with a fold id"),
        "unexpected missing-fold rejection: {error}"
    );

    let refit = fixture(true, false);
    let refit_state = Arc::new(CallState::default());
    *refit_state.emit_extra_fit_cv_partitions.lock().unwrap() = true;
    let mut refit_store = InMemoryArtifactStore::new();
    let refit_outcome = run(&refit, refit_state, &provider(&refit), &mut refit_store).unwrap();
    let final_output = &refit_outcome.outputs[0];
    assert_eq!(
        final_output.binding.prediction_source,
        PredictionSource::FinalRefit
    );
    assert!(
        !final_output.predictions.is_empty()
            && final_output.predictions.iter().all(|block| {
                block.partition == PredictionPartition::Final && block.fold_id.is_none()
            }),
        "FinalRefit remains Final-only even when FIT_CV emitted extra partitions"
    );
}

#[test]
fn stacking_cache_retention_and_discard_are_both_explicit() {
    let retained = fixture(false, true);
    let mut retained_store = InMemoryArtifactStore::new();
    let outcome = run(
        &retained,
        Arc::new(CallState::default()),
        &provider(&retained),
        &mut retained_store,
    )
    .unwrap();
    assert_eq!(outcome.execution_bundle.prediction_requirements.len(), 1);
    assert_eq!(outcome.execution_bundle.prediction_caches.len(), 1);
    let cache_record = &outcome.execution_bundle.prediction_caches[0];
    assert_eq!(
        outcome
            .portable_prediction_caches
            .as_ref()
            .unwrap()
            .caches
            .len(),
        1
    );
    let cache_payload = &outcome.portable_prediction_caches.as_ref().unwrap().caches[0];
    assert_eq!(
        cache_record.cache_namespace_fingerprints,
        cache_payload.cache_namespace_fingerprints
    );
    assert_eq!(
        cache_record.cache_namespace_fingerprints.len(),
        cache_record.blocks.len()
    );
    assert!(cache_record
        .cache_namespace_fingerprints
        .iter()
        .all(|fingerprint| fingerprint.len() == 64));
    let mut namespace_drift = cache_payload.clone();
    namespace_drift.cache_namespace_fingerprints[0] = "f".repeat(64);
    assert!(
        validate_prediction_cache_payload_matches_record(&namespace_drift, cache_record)
            .unwrap_err()
            .to_string()
            .contains("does not match cache record")
    );
    // The no-refit stacking outcome carries the full OOF triple (bundle
    // requirement + retained cache record + portable payload) for its in-closure
    // requires_oof edge, so REFIT replay is honestly self-contained.
    assert_eq!(outcome.replayable_phases, vec![Phase::Refit]);

    let mut discarded = fixture(true, true);
    discarded.request.options.artifacts.prediction_caches = PredictionCacheRetention::Discard;
    resign_request(&mut discarded.request);
    let projection = discarded.request.project().unwrap();
    discarded.influence = influence_manifest(&discarded.request, &projection, &discarded.relations);
    let mut discard_store = InMemoryArtifactStore::new();
    let discard_state = Arc::new(CallState::default());
    let error = run(
        &discarded,
        discard_state.clone(),
        &provider(&discarded),
        &mut discard_store,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("requires retained prediction caches"));
    assert_eq!(discard_state.total(), 0);
    assert!(discard_store.is_empty());
}

#[test]
fn explicit_selection_output_controls_multi_producer_ranking() {
    let fixture = fixture(false, true);
    let state = Arc::new(CallState::default());
    *state.score_auxiliary.lock().unwrap() = true;
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(&fixture, state, &provider(&fixture), &mut store).unwrap();
    assert_eq!(outcome.selection_output_id, "output:prediction");
    assert_eq!(outcome.selected_variant_id, fixture.preferred);
    assert_eq!(
        outcome
            .score_set
            .reports
            .iter()
            .filter(|report| {
                report.partition == PredictionPartition::Validation
                    && report
                        .fold_id
                        .as_ref()
                        .is_some_and(|fold| fold.as_str() == "avg")
            })
            .map(|report| report.producer_node.clone())
            .collect::<BTreeSet<_>>()
            .len(),
        2
    );
    let decision = outcome.execution_bundle.selections.values().next().unwrap();
    assert_eq!(decision.selected_candidate_id, fixture.preferred.as_str());
}

#[test]
fn native_training_materializes_operator_parameter_patches_before_execution() {
    let mut fixture = fixture(true, false);
    fixture.request.parameter_patches = vec![ParameterPatch {
        schema_version: PARAMETER_PATCH_SCHEMA_VERSION,
        node_id: node("model:base"),
        namespace: ParameterNamespace::Operator,
        path: vec!["patched_bias".to_string()],
        value: serde_json::json!(20),
    }];
    fixture.request.patch_policies = vec![NodePatchPolicy {
        node_id: node("model:base"),
        allowed_namespaces: [ParameterNamespace::Operator].into_iter().collect(),
    }];
    rebuild(&mut fixture);

    let state = Arc::new(CallState::default());
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(&fixture, state.clone(), &provider(&fixture), &mut store).unwrap();

    assert!(state
        .observed_model_patch_values
        .lock()
        .unwrap()
        .iter()
        .all(|value| value.as_ref() == Some(&serde_json::json!(20))));
    assert!(outcome.parameter_patches.iter().any(|patch| {
        patch.node_id.as_str() == "model:base"
            && patch.namespace == ParameterNamespace::Operator
            && patch.path == ["patched_bias".to_string()]
            && patch.value == serde_json::json!(20)
    }));
    let node_plan = &outcome.effective_plan.node_plans[&node("model:base")];
    assert_eq!(node_plan.params["patched_bias"], serde_json::json!(20));
    assert_eq!(
        node_plan.params_fingerprint,
        legacy_serde_fingerprint(&node_plan.params)
    );
    outcome.validate().unwrap();
}

#[test]
fn native_training_refuses_unexposed_or_structural_parameter_patches() {
    for (namespace, expected) in [
        (
            ParameterNamespace::Fit,
            "does not expose Fit parameter patches",
        ),
        (
            ParameterNamespace::Control,
            "does not expose Control parameter patches",
        ),
        (
            ParameterNamespace::Structural,
            "requires recompilation for structural parameter patches",
        ),
    ] {
        let mut fixture = fixture(true, false);
        fixture.request.parameter_patches = vec![ParameterPatch {
            schema_version: PARAMETER_PATCH_SCHEMA_VERSION,
            node_id: node("model:base"),
            namespace,
            path: vec!["patched_bias".to_string()],
            value: serde_json::json!(20),
        }];
        fixture.request.patch_policies = vec![NodePatchPolicy {
            node_id: node("model:base"),
            allowed_namespaces: [namespace].into_iter().collect(),
        }];
        rebuild(&mut fixture);
        let state = Arc::new(CallState::default());
        let mut store = InMemoryArtifactStore::new();
        let error = run(&fixture, state.clone(), &provider(&fixture), &mut store).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "unexpected error for {namespace:?}: {error}"
        );
        assert_eq!(state.total(), 0);
        assert!(store.is_empty());
    }
}

#[test]
fn outcome_rejects_selection_score_rank_and_producer_drift() {
    let fixture = fixture(false, false);
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(
        &fixture,
        Arc::new(CallState::default()),
        &provider(&fixture),
        &mut store,
    )
    .unwrap();

    let mut selected_score = outcome.clone();
    selected_score
        .execution_bundle
        .selections
        .values_mut()
        .next()
        .unwrap()
        .selected_score += 0.25;
    resign_outcome(&mut selected_score);
    assert!(selected_score.validate().is_err());

    let mut ranked_score = outcome.clone();
    ranked_score
        .execution_bundle
        .selections
        .values_mut()
        .next()
        .unwrap()
        .ranked_candidates[1]
        .score += 0.25;
    resign_outcome(&mut ranked_score);
    assert!(ranked_score.validate().is_err());

    let mut objective = outcome.clone();
    objective
        .execution_bundle
        .selections
        .values_mut()
        .next()
        .unwrap()
        .objective = MetricObjective::Maximize;
    resign_outcome(&mut objective);
    assert!(objective.validate().is_err());

    let mut mixed_output = outcome.clone();
    let source = mixed_output.outputs[0].predictions[0].clone();
    mixed_output.outputs[0]
        .observation_predictions
        .push(ObservationPredictionBlock {
            prediction_id: Some("prediction:resigned-observation".to_string()),
            producer_node: source.producer_node,
            producer_port: None,
            partition: source.partition,
            fold_id: source.fold_id,
            observation_ids: vec![ObservationId::new("observation:resigned").unwrap()],
            values: vec![vec![0.0]],
            weights: Vec::new(),
            target_names: source.target_names,
        });
    resign_outcome(&mut mixed_output);
    assert!(mixed_output.validate().is_err());

    let mut producer = outcome;
    producer
        .score_set
        .reports
        .iter_mut()
        .find(|report| {
            report
                .fold_id
                .as_ref()
                .is_some_and(|fold| fold.as_str() == "avg")
        })
        .unwrap()
        .producer_node = node("transform:snv");
    producer.execution_bundle.scores = Some(producer.score_set.clone());
    resign_outcome(&mut producer);
    assert!(producer.validate().is_err());
}

#[test]
fn divergent_selected_rerun_is_rejected() {
    let fixture = fixture(true, false);
    let state = Arc::new(CallState::default());
    *state.divergent_rerun.lock().unwrap() = true;
    let mut store = InMemoryArtifactStore::new();
    let error = run(&fixture, state, &provider(&fixture), &mut store).unwrap_err();
    assert!(error.to_string().contains("rerun diverged"));
    assert!(store.is_empty());
}

#[test]
fn late_output_failure_does_not_commit_artifacts() {
    let fixture = fixture(true, false);
    let state = Arc::new(CallState::default());
    *state.invalid_refit_output.lock().unwrap() = true;
    let mut store = InMemoryArtifactStore::new();
    assert!(run(&fixture, state.clone(), &provider(&fixture), &mut store).is_err());
    assert!(state.count(Phase::FitCv, "model:base") > 0);
    assert_eq!(state.count(Phase::Refit, "model:base"), 1);
    assert!(store.is_empty());
}

#[test]
fn provider_identity_and_relation_mismatches_fail_before_controllers() {
    let fixture = fixture(true, false);
    let state = Arc::new(CallState::default());
    let mut bad_identity = fixture.request.data_identities[0].clone();
    bad_identity.data_content_fingerprint = "f".repeat(64);
    bad_identity.identity_fingerprint = "0".repeat(64);
    bad_identity.identity_fingerprint = bad_identity.compute_fingerprint().unwrap();
    let bad_identity_provider = AttestedProvider {
        identity: bad_identity,
        relations: fixture.relations.clone(),
        contradictory_relations: None,
        omit_relations: false,
        next_handle: AtomicU64::new(0),
    };
    let mut store = InMemoryArtifactStore::new();
    assert!(run(&fixture, state.clone(), &bad_identity_provider, &mut store).is_err());
    assert_eq!(state.total(), 0);

    let mut contradictory = relations();
    contradictory.records.pop();
    let bad_relations_provider = AttestedProvider {
        identity: fixture.request.data_identities[0].clone(),
        relations: fixture.relations.clone(),
        contradictory_relations: Some(contradictory),
        omit_relations: false,
        next_handle: AtomicU64::new(0),
    };
    assert!(run(&fixture, state.clone(), &bad_relations_provider, &mut store).is_err());
    assert_eq!(state.total(), 0);

    let mut provider = provider(&fixture);
    provider.omit_relations = true;
    assert!(run(&fixture, state.clone(), &provider, &mut store).is_err());
    assert_eq!(state.total(), 0);
}

#[test]
fn native_training_enforces_controller_influence_capability_scopes() {
    for (capability, kind, requirements) in [
        (
            ControllerCapability::UsesEarlyStopping,
            TrainingInfluenceKind::EarlyStopping,
            early_stopping_requirements(),
        ),
        (
            ControllerCapability::UsesTrainingWeights,
            TrainingInfluenceKind::WeightingResampling,
            full_scope_requirements(TrainingInfluenceKind::WeightingResampling, "weighting"),
        ),
        (
            ControllerCapability::PerformsInternalTuning,
            TrainingInfluenceKind::HpoSelection,
            full_scope_requirements(TrainingInfluenceKind::HpoSelection, "internal_hpo"),
        ),
    ] {
        let mut fixture = fixture(true, false);
        let model_manifest = fixture
            .request
            .controller_manifests
            .iter_mut()
            .find(|manifest| manifest.operator_kind == NodeKind::Model)
            .unwrap();
        model_manifest.capabilities.insert(capability);
        if capability == ControllerCapability::UsesTrainingWeights {
            model_manifest
                .capabilities
                .insert(ControllerCapability::SupportsSampleWeights);
        }
        fixture.request.influence_requirements = requirements;
        rebuild(&mut fixture);

        let state = Arc::new(CallState::default());
        let mut store = InMemoryArtifactStore::new();
        let outcome = run(&fixture, state, &provider(&fixture), &mut store).unwrap();
        assert_eq!(
            outcome
                .training_influence
                .entries
                .iter()
                .filter(|entry| entry.kind == kind && entry.node_id.is_some())
                .count(),
            3,
            "capability {capability:?} must contribute every fold/refit scope"
        );
        outcome.validate().unwrap();
    }
}

#[test]
fn native_training_rejects_missing_or_leaking_controller_influence_before_controllers() {
    assert_preflight_rejected(fixture(true, false), |request| {
        request
            .controller_manifests
            .iter_mut()
            .find(|manifest| manifest.operator_kind == NodeKind::Model)
            .unwrap()
            .capabilities
            .insert(ControllerCapability::UsesEarlyStopping);
    });

    assert_preflight_rejected(fixture(true, false), |request| {
        request
            .controller_manifests
            .iter_mut()
            .find(|manifest| manifest.operator_kind == NodeKind::Model)
            .unwrap()
            .capabilities
            .insert(ControllerCapability::UsesEarlyStopping);
        request.influence_requirements = early_stopping_requirements();
        request.influence_requirements[0].physical_sample_ids = vec![sample("sample:1")];
    });
}

#[test]
fn native_training_persists_runtime_derived_influence_evidence() {
    let fixture = fixture(true, false);
    let projection = fixture.request.project().unwrap();
    let expected = TrainingInfluenceManifest::derive_for_projection(
        &projection,
        &fixture.request,
        &fixture.relations,
    )
    .unwrap();
    let state = Arc::new(CallState::default());
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(&fixture, state, &provider(&fixture), &mut store).unwrap();

    assert_eq!(outcome.training_influence, expected);
    assert!(outcome.training_influence.entries.iter().any(|entry| {
        entry.kind == TrainingInfluenceKind::HpoSelection
            && entry.node_id.is_none()
            && entry.scope_id.starts_with("select:")
            && entry.group_ids
                == vec![
                    GroupId::new("group:0").unwrap(),
                    GroupId::new("group:1").unwrap(),
                ]
    }));
    assert!(outcome.training_influence.entries.iter().all(|entry| {
        let unique_groups = entry.group_ids.iter().collect::<BTreeSet<_>>();
        !entry.physical_sample_ids.is_empty()
            && unique_groups.len() == entry.group_ids.len()
            && entry.group_ids.windows(2).all(|pair| pair[0] < pair[1])
    }));
    outcome.validate().unwrap();
}

#[test]
fn classification_selection_is_native_when_columns_are_coherent() {
    let mut fixture = fixture(false, false);
    fixture.request.options.selection.metric.name = "balanced_accuracy".to_string();
    fixture.request.options.selection.metric.objective = MetricObjective::Maximize;
    fixture.request.options.outputs[0].prediction_kind = PredictionKind::ClassLabel;
    fixture.request.options.outputs[0].class_labels = vec![vec!["0".to_string(), "1".to_string()]];
    resign_request(&mut fixture.request);
    let projection = fixture.request.project().unwrap();
    fixture.preferred = projection.plan.variants[0].variant_id.clone();
    fixture.influence = influence_manifest(&fixture.request, &projection, &fixture.relations);
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(
        &fixture,
        Arc::new(CallState::default()),
        &provider(&fixture),
        &mut store,
    )
    .unwrap();
    assert_eq!(
        outcome.score_set.selection_metric.as_deref(),
        Some("balanced_accuracy")
    );
    assert_eq!(
        outcome.outputs[0].binding.prediction_kind,
        PredictionKind::ClassLabel
    );
}

#[test]
fn parallel_threads_matches_sequential_selection_and_lineage() {
    let sequential = fixture(false, false);
    let mut sequential_store = InMemoryArtifactStore::new();
    let sequential_outcome = run(
        &sequential,
        Arc::new(CallState::default()),
        &provider(&sequential),
        &mut sequential_store,
    )
    .unwrap();

    let mut parallel = fixture(false, false);
    parallel.request.options.scheduler.kind = TrainingSchedulerKind::Parallel;
    parallel.request.options.scheduler.backend = Some(TrainingSchedulerBackend::Threads);
    parallel.request.options.scheduler.workers = 2;
    parallel.request.options.resources.cpu_threads = 2;
    rebuild(&mut parallel);
    let mut parallel_store = InMemoryArtifactStore::new();
    let parallel_outcome = run(
        &parallel,
        Arc::new(CallState::default()),
        &provider(&parallel),
        &mut parallel_store,
    )
    .unwrap();
    parallel_outcome.validate().unwrap();
    assert_eq!(
        parallel_outcome.selected_variant_id,
        sequential_outcome.selected_variant_id
    );
    assert_eq!(parallel_outcome.score_set, sequential_outcome.score_set);
    assert_eq!(parallel_outcome.outputs, sequential_outcome.outputs);
    assert_eq!(parallel_outcome.lineage, sequential_outcome.lineage);
}

#[test]
fn unsupported_options_are_never_silently_ignored() {
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.resources.memory_bytes = Some(1024)
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.resources.gpu_devices = vec!["gpu:0".to_string()]
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.resources.wall_time_ms = Some(1000)
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.resources.cpu_threads = 2
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.artifacts.cv_artifacts = CvArtifactRetention::MetadataOnly
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.artifacts.fitted_artifacts = FittedArtifactMode::PortableRequired
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.refit_strategy = Some(RefitStrategy::RefitEnsemble)
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.selection.reduction_id = Some("reduction:test".to_string())
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.selection.stacking_fit_contract = Some(StackingFitContract {
            meta_training_features: MetaTrainingFeatures::Oof,
            inference_features: InferenceFeatures::RefitBasePredictions,
            selection_protocol: SelectionProtocol::Nested,
            meta_row_domain: MetaRowDomain::Sample,
            final_reduction_id: None,
            unsafe_allow_reuse_oof: false,
        })
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.selection.require_finite = false
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.selection.evaluation_scope = Some(EvaluationScope::Holdout)
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.selection.required_metric_level = Some(PredictionLevel::Group)
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.selection.refit_slot_plan = Some(RefitSlotPlan {
            strategy: RefitStrategy::RefitOne,
            selection_level: PredictionLevel::Group,
            member_count: 1,
            selection_metric: request.options.selection.metric.clone(),
            reduction_id: None,
        })
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.scheduler.kind = TrainingSchedulerKind::Parallel;
        request.options.scheduler.backend = Some(TrainingSchedulerBackend::Processes);
        request.options.scheduler.workers = 2;
        request.options.resources.cpu_threads = 2;
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.parameter_patches.push(ParameterPatch {
            schema_version: PARAMETER_PATCH_SCHEMA_VERSION,
            node_id: node("model:base"),
            namespace: ParameterNamespace::Operator,
            path: vec!["n_estimators".to_string()],
            value: serde_json::json!(20),
        });
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.outputs[0].prediction_kind = PredictionKind::ClassProbability;
        request.options.outputs[0].output_order = OutputOrder::TargetMajorClassMinor;
        request.options.outputs[0].class_labels = vec![vec!["0".to_string(), "1".to_string()]];
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.outputs[0].prediction_kind = PredictionKind::DecisionScore;
    });
    assert_preflight_rejected(fixture(true, false), |request| {
        request.options.selection.metric.name = "accuracy".to_string();
        request.options.selection.metric.objective = MetricObjective::Maximize;
    });
}

#[test]
fn partial_predictor_closure_and_legacy_multi_prediction_ports_fail_closed() {
    let mut partial = fixture(true, false);
    let mut unused = partial.request.graph.nodes[0].clone();
    unused.id = node("transform:unused");
    unused.seed_label = Some("unused".to_string());
    partial.request.graph.nodes.push(unused);
    rebuild(&mut partial);
    let partial_state = Arc::new(CallState::default());
    let mut store = InMemoryArtifactStore::new();
    let error = run(
        &partial,
        partial_state.clone(),
        &provider(&partial),
        &mut store,
    )
    .unwrap_err();
    assert!(error.to_string().contains("predictor closure"));
    assert_eq!(partial_state.total(), 0);

    let mut multi_port = fixture(true, false);
    let mut extra = multi_port
        .request
        .graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .outputs[0]
        .clone();
    extra.name = "probability".to_string();
    multi_port
        .request
        .graph
        .nodes
        .iter_mut()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .outputs
        .push(extra.clone());
    multi_port
        .request
        .controller_manifests
        .iter_mut()
        .find(|manifest| manifest.controller_id.as_str() == "controller:model.mock")
        .unwrap()
        .output_ports
        .push(extra);
    multi_port.request.options.outputs[0].port_name = Some("oof".to_string());
    resign_request(&mut multi_port.request);
    let multi_state = Arc::new(CallState::default());
    let error = run(
        &multi_port,
        multi_state.clone(),
        &provider(&multi_port),
        &mut store,
    )
    .unwrap_err();
    assert!(error.to_string().contains("without producer_port"));
    assert!(multi_state.total() > 0);

    let mut upstream_multi = fixture(true, true);
    let mut extra = upstream_multi
        .request
        .graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "transform:snv")
        .unwrap()
        .ports
        .outputs
        .iter()
        .find(|port| port.kind == PortKind::Prediction)
        .unwrap()
        .clone();
    extra.name = "oof_aux_second".to_string();
    upstream_multi
        .request
        .graph
        .nodes
        .iter_mut()
        .find(|node| node.id.as_str() == "transform:snv")
        .unwrap()
        .ports
        .outputs
        .push(extra.clone());
    upstream_multi
        .request
        .controller_manifests
        .iter_mut()
        .find(|manifest| manifest.controller_id.as_str() == "controller:transform.mock")
        .unwrap()
        .output_ports
        .push(extra);
    rebuild(&mut upstream_multi);
    let upstream_state = Arc::new(CallState::default());
    let error = run(
        &upstream_multi,
        upstream_state.clone(),
        &provider(&upstream_multi),
        &mut store,
    )
    .unwrap_err();
    assert!(error.to_string().contains("without producer_port"));
    assert!(upstream_state.total() > 0);
}

#[test]
fn explicit_multi_prediction_port_output_binds_requested_port_only() {
    let mut fixture = fixture(true, false);
    let mut extra = fixture
        .request
        .graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .outputs[0]
        .clone();
    extra.name = "probability".to_string();
    fixture
        .request
        .graph
        .nodes
        .iter_mut()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .outputs
        .push(extra.clone());
    fixture
        .request
        .controller_manifests
        .iter_mut()
        .find(|manifest| manifest.controller_id.as_str() == "controller:model.mock")
        .unwrap()
        .output_ports
        .push(extra);
    fixture.request.options.outputs[0].port_name = Some("oof".to_string());
    rebuild(&mut fixture);

    let state = Arc::new(CallState::default());
    *state.emit_explicit_model_ports.lock().unwrap() = true;
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(&fixture, state, &provider(&fixture), &mut store).unwrap();
    assert_eq!(outcome.outputs.len(), 1);
    let output = &outcome.outputs[0];
    assert_eq!(output.binding.node_id.as_str(), "model:base");
    assert_eq!(output.binding.port_name, "oof");
    assert!(output.observation_predictions.is_empty());
    assert!(output.aggregated_predictions.is_empty());
    assert!(!output.predictions.is_empty());
    assert!(output.predictions.iter().all(|block| {
        block.producer_node.as_str() == "model:base"
            && block.producer_port.as_deref() == Some("oof")
            && block.partition == PredictionPartition::Final
            && block.fold_id.is_none()
    }));
    assert!(outcome
        .score_set
        .reports
        .iter()
        .any(|report| report.producer_node.as_str() == "model:base"
            && report.producer_port.as_deref() == Some("probability")));
}

#[test]
fn identifiers_controllers_diagnostics_and_store_are_prevalidated() {
    let fixture = fixture(true, false);
    let provider = provider(&fixture);

    let state = Arc::new(CallState::default());
    let mut store = InMemoryArtifactStore::new();
    assert!(run_custom(
        &fixture,
        state.clone(),
        &provider,
        &mut store,
        "not a portable id",
        BTreeMap::new(),
        true,
    )
    .is_err());
    assert_eq!(state.total(), 0);

    assert!(run_custom(
        &fixture,
        state.clone(),
        &provider,
        &mut store,
        "outcome:test.preflight",
        BTreeMap::from([(
            "bad".to_string(),
            serde_json::json!({"handle": 7, "owner_controller": "controller:model.mock"}),
        )]),
        true,
    )
    .is_err());
    assert_eq!(state.total(), 0);

    assert!(run_custom(
        &fixture,
        state.clone(),
        &provider,
        &mut store,
        "outcome:test.preflight",
        BTreeMap::new(),
        false,
    )
    .is_err());
    assert_eq!(state.total(), 0);

    let record = RefitArtifactRecord {
        node_id: node("model:base"),
        controller_id: ControllerId::new("controller:model.mock").unwrap(),
        artifact: ArtifactRef {
            id: ArtifactId::new("artifact:preexisting").unwrap(),
            kind: "test".to_string(),
            controller_id: ControllerId::new("controller:model.mock").unwrap(),
            backend: None,
            uri: None,
            content_fingerprint: None,
            size_bytes: Some(1),
            plugin: None,
            plugin_version: None,
        },
        params_fingerprint: "a".repeat(64),
        training_loss_fingerprint: None,
        data_requirement_keys: Vec::new(),
        prediction_requirement_keys: Vec::new(),
    };
    store
        .register(
            &record,
            HandleRef {
                handle: 1,
                kind: HandleKind::Artifact,
                owner_controller: ControllerId::new("controller:model.mock").unwrap(),
            },
        )
        .unwrap();
    assert!(run_custom(
        &fixture,
        state.clone(),
        &provider,
        &mut store,
        "outcome:test.preflight",
        BTreeMap::new(),
        true,
    )
    .is_err());
    assert_eq!(state.total(), 0);
}

fn set_transform_manifest(fixture: &mut Fixture, mutate: impl Fn(&mut ControllerManifest)) {
    for manifest in &mut fixture.request.controller_manifests {
        if manifest.controller_id.as_str() == "controller:transform.mock" {
            mutate(manifest);
        }
    }
    rebuild(fixture);
}

// A stateless transform whose `artifact_policy` is `ReplayRequired` still carries
// no reloadable inference state: retained state is required for
// `Stateful || EmitsArtifacts` only, never for `ReplayRequired` or `fit_scope`.
// It therefore has no refit artifact yet keeps the completed-refit outcome
// PREDICT-replayable — a real integration proof, not a pure fact-table case.
#[test]
fn completed_refit_stateless_replay_required_node_needs_no_artifact() {
    let mut fixture = fixture(true, false);
    set_transform_manifest(&mut fixture, |manifest| {
        manifest.artifact_policy = ArtifactPolicy::ReplayRequired;
    });
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(
        &fixture,
        Arc::new(CallState::default()),
        &provider(&fixture),
        &mut store,
    )
    .unwrap();
    outcome.validate().unwrap();
    let transform = &outcome.effective_plan.node_plans[&node("transform:snv")];
    assert_eq!(transform.artifact_policy, ArtifactPolicy::ReplayRequired);
    assert!(!transform
        .controller_capabilities
        .contains(&ControllerCapability::Stateful));
    assert!(!transform
        .controller_capabilities
        .contains(&ControllerCapability::EmitsArtifacts));
    assert!(!outcome
        .execution_bundle
        .refit_artifacts
        .iter()
        .any(|artifact| artifact.node_id.as_str() == "transform:snv"));
    assert_eq!(outcome.replayable_phases, vec![Phase::Predict]);
}

// A `Stateful` node that emits no artifact requires retained inference state but
// has no retained artifact (only `EmitsArtifacts` nodes produce refit artifacts),
// so the honest completed-refit answer is [] — never a false PREDICT.
#[test]
fn completed_refit_stateful_non_emitter_without_artifact_derives_empty() {
    let mut fixture = fixture(true, false);
    set_transform_manifest(&mut fixture, |manifest| {
        manifest.capabilities.insert(ControllerCapability::Stateful);
    });
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(
        &fixture,
        Arc::new(CallState::default()),
        &provider(&fixture),
        &mut store,
    )
    .unwrap();
    outcome.validate().unwrap();
    let transform = &outcome.effective_plan.node_plans[&node("transform:snv")];
    assert!(transform
        .controller_capabilities
        .contains(&ControllerCapability::Stateful));
    assert!(!outcome
        .execution_bundle
        .refit_artifacts
        .iter()
        .any(|artifact| artifact.node_id.as_str() == "transform:snv"));
    assert!(outcome.replayable_phases.is_empty());
}

// The completed-refit outcome advertises exactly [PREDICT]; a re-signed outcome
// forging a stronger or weaker replay claim is rejected by re-derivation.
#[test]
fn refit_outcome_rejects_forged_replay_claims_even_when_resigned() {
    let fixture = fixture(true, false);
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(
        &fixture,
        Arc::new(CallState::default()),
        &provider(&fixture),
        &mut store,
    )
    .unwrap();
    assert_eq!(outcome.replayable_phases, vec![Phase::Predict]);

    // Advertising EXPLAIN (model.mock does not support it) is refused.
    let mut explain_claim = outcome.clone();
    explain_claim.replayable_phases = vec![Phase::Predict, Phase::Explain];
    resign_outcome(&mut explain_claim);
    let error = explain_claim.validate().unwrap_err();
    assert!(
        error.to_string().contains("replayable_phases do not match"),
        "{error}"
    );

    // A completed refit re-advertising REFIT is refused.
    let mut refit_claim = outcome.clone();
    refit_claim.replayable_phases = vec![Phase::Refit];
    resign_outcome(&mut refit_claim);
    let error = refit_claim.validate().unwrap_err();
    assert!(
        error.to_string().contains("replayable_phases do not match"),
        "{error}"
    );

    // Dropping the honest PREDICT is refused too — [] is not honest here.
    let mut empty_claim = outcome;
    empty_claim.replayable_phases = Vec::new();
    resign_outcome(&mut empty_claim);
    let error = empty_claim.validate().unwrap_err();
    assert!(
        error.to_string().contains("replayable_phases do not match"),
        "{error}"
    );
}

// Re-signing the outer outcome fingerprint cannot launder plan topology or
// adjacency drift: the embedded ExecutionPlan is independently re-validated
// (canonical topological order and edge-derived input/output adjacency).
#[test]
fn re_signed_plan_topology_and_adjacency_drift_are_rejected() {
    let fixture = fixture(true, false);
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(
        &fixture,
        Arc::new(CallState::default()),
        &provider(&fixture),
        &mut store,
    )
    .unwrap();

    let mut topology = outcome.clone();
    topology
        .effective_plan
        .graph_plan
        .topological_order
        .reverse();
    resign_outcome(&mut topology);
    let error = topology.validate().unwrap_err();
    assert!(error.to_string().contains("topological"), "{error}");

    let mut adjacency = outcome;
    adjacency
        .effective_plan
        .node_plans
        .get_mut(&node("model:base"))
        .unwrap()
        .input_nodes
        .clear();
    resign_outcome(&mut adjacency);
    let error = adjacency.validate().unwrap_err();
    assert!(
        error.to_string().contains("input/output adjacency"),
        "{error}"
    );
}

#[test]
fn portable_package_independently_requires_predict_replayability() {
    let mut package: PortablePredictorPackage = serde_json::from_str(PACKAGE_FIXTURE).unwrap();
    package.validate().unwrap();

    let controller_id = ControllerId::new("controller:augmentation.mock").unwrap();
    package
        .effective_plan
        .controller_manifests
        .get_mut(&controller_id)
        .unwrap()
        .supported_phases
        .remove(&Phase::Predict);
    for node_plan in package.effective_plan.node_plans.values_mut() {
        if node_plan.controller_id == controller_id {
            node_plan.supported_phases.remove(&Phase::Predict);
        }
    }
    package.effective_plan.controller_fingerprint =
        legacy_serde_fingerprint(&package.effective_plan.controller_manifests);
    package.execution_bundle.controller_fingerprint =
        package.effective_plan.controller_fingerprint.clone();
    package.template.controller_manifests = package.effective_plan.controller_manifests.clone();
    package.template.template_fingerprint = "0".repeat(64);
    package.template.template_fingerprint = package.template.compute_fingerprint().unwrap();
    package.training_outcome.effective_plan_fingerprint =
        typed_fingerprint(&package.effective_plan);
    package.training_outcome.execution_bundle_fingerprint =
        typed_fingerprint(&package.execution_bundle);
    package.package_fingerprint = "0".repeat(64);
    package.package_fingerprint = package.compute_fingerprint().unwrap();

    let error = package.validate().unwrap_err();
    assert!(error.to_string().contains("PREDICT-replayable"), "{error}");
}

#[test]
fn d8_training_outcome_exports_loadable_host_sidecar_package() {
    let fixture = fixture(true, false);
    let state = Arc::new(CallState::default());
    let mut store = InMemoryArtifactStore::new();
    let outcome = run(&fixture, state, &provider(&fixture), &mut store).unwrap();
    assert_eq!(outcome.refit.status, TrainingRefitStatus::Completed);
    assert!(!outcome.execution_bundle.refit_artifacts.is_empty());

    let package = outcome
        .to_portable_predictor_package(
            "predictor:package.d8.host_sidecar",
            FittedArtifactMode::AllowHostSidecar,
            ArtifactLoadMode::HostSidecar,
        )
        .unwrap();
    package.validate().unwrap();
    assert_eq!(
        package.artifact_bindings.len(),
        outcome.execution_bundle.refit_artifacts.len()
    );
    assert!(package
        .artifact_bindings
        .iter()
        .all(|binding| binding.load_mode == ArtifactLoadMode::HostSidecar));
    assert_eq!(
        package.training_outcome.outcome_fingerprint,
        outcome.outcome_fingerprint
    );
    assert_eq!(
        package
            .output_bindings
            .iter()
            .map(|binding| binding.binding_fingerprint.clone())
            .collect::<Vec<_>>(),
        outcome
            .outputs
            .iter()
            .map(|output| output.binding.binding_fingerprint.clone())
            .collect::<Vec<_>>()
    );

    let json = serde_json::to_string(&package).unwrap();
    let parsed = PortablePredictorPackage::from_json(&json).unwrap();
    let loaded = parsed
        .clone()
        .load_with(|record| Ok(format!("sidecar:{}", record.artifact.id)))
        .unwrap();
    for binding in &parsed.artifact_bindings {
        assert_eq!(
            loaded.artifact(&binding.artifact_id).unwrap(),
            &format!("sidecar:{}", binding.artifact_id)
        );
    }

    let mut stale = parsed;
    stale
        .execution_bundle
        .metadata
        .insert("stale_bundle".to_string(), serde_json::json!(true));
    stale.package_fingerprint = "0".repeat(64);
    stale.package_fingerprint = stale.compute_fingerprint().unwrap();
    let error = stale.validate().unwrap_err();
    assert!(
        error.to_string().contains("execution bundle content"),
        "{error}"
    );
}

#[test]
fn d8_loaded_predictor_replays_predict_without_source_training_outcome() {
    let mut fixture = fixture(true, false);
    add_model_probability_port(&mut fixture);
    add_explain_support(&mut fixture);
    let state = Arc::new(CallState::default());
    *state.emit_explicit_model_ports.lock().unwrap() = true;
    let mut store = InMemoryArtifactStore::new();
    let source = run(&fixture, state.clone(), &provider(&fixture), &mut store)
        .expect("source training outcome");
    let package = source
        .to_portable_predictor_package(
            "predictor:package.d8.stateless",
            FittedArtifactMode::AllowHostSidecar,
            ArtifactLoadMode::HostSidecar,
        )
        .unwrap();
    let loaded = package
        .clone()
        .load_with(|record| {
            store
                .get(&record.artifact.id)
                .map(|handle_record| handle_record.handle.clone())
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "missing sidecar handle for `{}`",
                        record.artifact.id
                    ))
                })
        })
        .unwrap();

    let current_relation = "e".repeat(64);
    let envelopes = replay_envelopes_with_relation(&source, &current_relation);
    let request = replay_request(&source, Phase::Predict);
    let controllers = controllers(&fixture, state.clone(), true);
    let replay = execute_loaded_predictor_replay(LoadedPredictorReplayInput {
        predictor: &loaded,
        request: &request,
        outcome_id: "replay:loaded.predict.outcome".to_string(),
        run_id: RunId::new("run:loaded.predict").unwrap(),
        controllers: &controllers,
        data_provider: &provider(&fixture),
        data_envelopes: &envelopes,
        warnings: Vec::new(),
        diagnostics: BTreeMap::from([("loaded_package".to_string(), serde_json::json!(true))]),
    })
    .expect("loaded package replay");

    assert_eq!(replay.phase, Phase::Predict);
    assert_eq!(replay.source_training_outcome, package.training_outcome);
    assert_eq!(
        replay.replay_request_fingerprint,
        request.request_fingerprint
    );
    assert_eq!(replay.outputs.len(), request.output_binding_ids.len());
    assert!(replay.explanations.is_empty());
    assert!(replay
        .input_data_identities
        .iter()
        .all(|identity| identity.relation_fingerprint == current_relation));
    assert!(replay.outputs.iter().all(|output| {
        output.schema_version == Some(BOUND_TRAINING_OUTPUT_SCHEMA_VERSION)
            && !output.predictions.is_empty()
            && output.predictions.iter().all(|block| {
                block.partition == PredictionPartition::Final
                    && block.fold_id.is_none()
                    && block.producer_port.as_deref() == Some(output.binding.port_name.as_str())
            })
    }));
    replay
        .validate_against_package(loaded.package(), &request)
        .unwrap();
    assert!(state.count(Phase::Predict, "model:base") > 0);

    let mut explain_request = request.clone();
    explain_request.phase = Phase::Explain;
    explain_request.request_fingerprint = "0".repeat(64);
    explain_request.request_fingerprint = explain_request.compute_fingerprint().unwrap();
    let replay = execute_loaded_predictor_replay(LoadedPredictorReplayInput {
        predictor: &loaded,
        request: &explain_request,
        outcome_id: "replay:loaded.explain.outcome".to_string(),
        run_id: RunId::new("run:loaded.explain").unwrap(),
        controllers: &controllers,
        data_provider: &provider(&fixture),
        data_envelopes: &envelopes,
        warnings: Vec::new(),
        diagnostics: BTreeMap::new(),
    })
    .expect("loaded package explain replay");
    assert_eq!(replay.phase, Phase::Explain);
    assert!(replay.outputs.is_empty());
    assert_eq!(replay.explanation_block_count, replay.explanations.len());
    assert!(replay.explanations.iter().any(|block| {
        block.producer_node.as_str() == "model:base"
            && block.producer_port.as_deref() == Some("oof")
            && block.method == "fixture_explain"
    }));
    replay
        .validate_against_package(loaded.package(), &explain_request)
        .unwrap();
    assert!(state.count(Phase::Explain, "model:base") > 0);
}

fn positional_struct(
    value: &serde_json::Value,
    fields: &[(&str, serde_json::Value)],
) -> serde_json::Value {
    let object = value.as_object().expect("fixture struct is an object");
    serde_json::Value::Array(
        fields
            .iter()
            .map(|(name, default)| {
                object
                    .get(*name)
                    .cloned()
                    .unwrap_or_else(|| default.clone())
            })
            .collect(),
    )
}

#[test]
fn standalone_contract_readers_reject_serde_positional_struct_wires() {
    let package: serde_json::Value = serde_json::from_str(PACKAGE_FIXTURE).unwrap();

    let graph = &package["template"]["graph"];
    let graph_sequence = positional_struct(
        graph,
        &[
            ("id", serde_json::Value::Null),
            ("interface", serde_json::json!({})),
            ("nodes", serde_json::json!([])),
            ("edges", serde_json::json!([])),
            ("search_space_fingerprint", serde_json::Value::Null),
            ("metadata", serde_json::json!({})),
        ],
    );
    let permissive_graph: GraphSpec = serde_json::from_value(graph_sequence.clone()).unwrap();
    permissive_graph.validate().unwrap();
    assert!(GraphSpec::from_json(&serde_json::to_string(&graph_sequence).unwrap()).is_err());

    let campaign = &package["template"]["campaign"];
    let campaign_sequence = positional_struct(
        campaign,
        &[
            ("id", serde_json::Value::Null),
            ("root_seed", serde_json::Value::Null),
            ("leakage_policy", serde_json::json!({})),
            ("aggregation_policy", serde_json::json!({})),
            ("split_invocation", serde_json::Value::Null),
            ("generation", serde_json::json!({})),
            ("shape_plans", serde_json::json!({})),
            ("data_bindings", serde_json::json!({})),
            ("branch_view_plans", serde_json::json!([])),
            ("inner_cv", serde_json::Value::Null),
            ("metadata", serde_json::json!({})),
        ],
    );
    let permissive_campaign: CampaignSpec =
        serde_json::from_value(campaign_sequence.clone()).unwrap();
    permissive_campaign.validate().unwrap();
    assert!(CampaignSpec::from_json(&serde_json::to_string(&campaign_sequence).unwrap()).is_err());

    let bundle = &package["execution_bundle"];
    let bundle_fields = [
        ("bundle_id", serde_json::Value::Null),
        ("schema_version", serde_json::json!(1)),
        ("plan_id", serde_json::Value::Null),
        ("graph_fingerprint", serde_json::Value::Null),
        ("campaign_fingerprint", serde_json::Value::Null),
        ("controller_fingerprint", serde_json::Value::Null),
        ("selected_variant_id", serde_json::Value::Null),
        ("selections", serde_json::json!({})),
        ("refit_artifacts", serde_json::json!([])),
        ("prediction_requirements", serde_json::json!([])),
        ("prediction_caches", serde_json::json!([])),
        ("scores", serde_json::Value::Null),
        ("data_requirements", serde_json::json!([])),
        ("unsafe_flags", serde_json::json!([])),
        ("metadata", serde_json::json!({})),
    ];
    let bundle_sequence = positional_struct(bundle, &bundle_fields);
    let permissive_bundle: ExecutionBundle =
        serde_json::from_value(bundle_sequence.clone()).unwrap();
    permissive_bundle.validate().unwrap();
    assert!(ExecutionBundle::from_json(&serde_json::to_string(&bundle_sequence).unwrap()).is_err());

    let mut nested_bundle = bundle.clone();
    let artifact = nested_bundle["refit_artifacts"][0].clone();
    nested_bundle["refit_artifacts"][0] = positional_struct(
        &artifact,
        &[
            ("node_id", serde_json::Value::Null),
            ("controller_id", serde_json::Value::Null),
            ("artifact", serde_json::Value::Null),
            ("params_fingerprint", serde_json::Value::Null),
            ("training_loss_fingerprint", serde_json::Value::Null),
            ("data_requirement_keys", serde_json::json!([])),
            ("prediction_requirement_keys", serde_json::json!([])),
        ],
    );
    let permissive_nested: ExecutionBundle = serde_json::from_value(nested_bundle.clone()).unwrap();
    permissive_nested.validate().unwrap();
    assert!(ExecutionBundle::from_json(&serde_json::to_string(&nested_bundle).unwrap()).is_err());
}
