use dag_ml_core::*;
use sha2::{Digest,Sha256};
use std::collections::{BTreeMap,BTreeSet};
use std::sync::{Arc,Mutex};
use std::sync::atomic::{AtomicU64,Ordering};
#[derive(Default)]
struct CallState {
    calls: Mutex<Vec<(Phase, NodeId)>>,
    scoped_calls: Mutex<Vec<(Phase, NodeId, Option<FoldId>)>>,
    observed_resources: Mutex<Vec<Option<TrainingResourceLimits>>>,
    fit_counts: Mutex<BTreeMap<VariantId, usize>>,
    next_handle: AtomicU64,
    preferred: Mutex<Option<VariantId>>,
    divergent_rerun: Mutex<bool>,
    invalid_refit_output: Mutex<bool>,
    artifact_only_refit: Mutex<bool>,
    score_auxiliary: Mutex<bool>,
    emit_extra_fit_cv_partitions: Mutex<bool>,
    emit_explicit_model_ports: Mutex<bool>,
    predict_sample_ids: Mutex<Option<Vec<SampleId>>>,
    scoped_validation_samples: Mutex<BTreeMap<FoldId, Vec<SampleId>>>,
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
            .observed_resources
            .lock()
            .unwrap()
            .push(task.resources.clone());
        self.state
            .calls
            .lock()
            .unwrap()
            .push((task.phase, task.node_plan.node_id.clone()));
        self.state.scoped_calls.lock().unwrap().push((
            task.phase,
            task.node_plan.node_id.clone(),
            task.fold_id.clone(),
        ));
        let is_model = task.node_plan.node_id.as_str().starts_with("model:");
        if is_model {
            self.state
                .observed_model_patch_values
                .lock()
                .unwrap()
                .push(task.node_plan.params.get("patched_bias").cloned());
        }
        let scoped_samples = task.fold_id.as_ref().and_then(|fold| {
            self.state
                .scoped_validation_samples
                .lock()
                .unwrap()
                .get(fold)
                .cloned()
        });
        let sample_ids =
            scoped_samples.unwrap_or_else(|| match task.fold_id.as_ref().map(FoldId::as_str) {
                Some("fold:0") => vec![sample("sample:1"), sample("sample:2")],
                Some("fold:1") => vec![sample("sample:3"), sample("sample:4")],
                None if task.phase == Phase::Predict => self
                    .state
                    .predict_sample_ids
                    .lock()
                    .unwrap()
                    .clone()
                    .unwrap_or_else(|| {
                        (1..=4)
                            .map(|index| sample(&format!("sample:{index}")))
                            .collect()
                    }),
                _ => (1..=4)
                    .map(|index| sample(&format!("sample:{index}")))
                    .collect(),
            });

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
            is_model && task.node_plan.params.contains_key("second_targets") && *self.state.emit_explicit_model_ports.lock().unwrap();
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
                producer_port: is_model.then(|| "oof".to_string()),
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
            sibling.target_names = task.node_plan.params.get("second_targets").map(|v| serde_json::from_value(v.clone()).unwrap()).unwrap_or_else(|| vec!["fat".into()]);
            sibling.values = sibling
                .values
                .iter()
                .map(|row| row.iter().map(|value| value + 100.0).collect())
                .collect();
            sibling.values = sample_ids.iter().map(|_| vec![100.0 + value;sibling.target_names.len()]).collect();
            predictions.push(sibling);
        }
        if is_model && task.phase == Phase::Refit && *self.state.artifact_only_refit.lock().unwrap()
        {
            predictions.clear();
        }
        if self.emits_predictions
            && task.phase == Phase::FitCv
            && *self.state.emit_extra_fit_cv_partitions.lock().unwrap()
        {
            let validation = predictions
                .first()
                .expect("FIT_CV prediction controller emits Validation")
                .clone();
            let train_sample_ids = task
                .data_views
                .get("data:x")
                .and_then(|view| view.sample_ids.clone())
                .expect("FIT_CV train block uses its attested fold-train view");
            let train_values = vec![vec![value]; train_sample_ids.len()];
            predictions.extend([
                PredictionBlock {
                    prediction_id: Some(format!(
                        "prediction:{}:FIT_CV:train:{}",
                        task.node_plan.node_id,
                        task.fold_id.as_ref().map(FoldId::as_str).unwrap_or("full")
                    )),
                    partition: PredictionPartition::Train,
                    sample_ids: train_sample_ids,
                    values: train_values,
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
                producer_port: is_model.then(|| "oof".to_string()),
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
                validity_masks: None,
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
        let mut regression_targets = regression_targets;
        if explicit_model_ports && task.phase == Phase::FitCv {
            let mut truth = regression_targets[0].clone();
            truth.target_names = predictions[1].target_names.clone();
            truth.values = sample_ids.iter().map(|_| vec![100.0;truth.target_names.len()]).collect();
            if truth.target_names != regression_targets[0].target_names { regression_targets.push(truth); }
        }
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
                abi_major: None,
                abi_min_minor: None,
                native_predictor_descriptor: None,
                native_estimator_descriptor: None,
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
        let result = NodeResult {
            schema_version: None,
            classification_probabilities: Vec::new(),
            consumed_data_views: std::collections::BTreeMap::new(),
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
                early_stopping_records: Vec::new(),
                target_domain: None,
            },
        };
        log_event(serde_json::json!({"event":"invoke","task":task}));
        Ok(result)
    }
}

struct Fixture {
    request: TrainingRequest,
    relations: SampleRelationSet,
    influence: TrainingInfluenceManifest,
    preferred: VariantId,
}

fn fixture(refit: bool, stacking: bool) -> Fixture {
    let graph: GraphSpec =
        serde_json::from_str(include_str!("../inputs/minimal_graph.json")).unwrap();
    let mut campaign: CampaignSpec = serde_json::from_str(include_str!(
        "../inputs/campaign_oof_generation.json"
    ))
    .unwrap();
    let mut controller_manifests: Vec<ControllerManifest> =
        serde_json::from_str(include_str!("../inputs/controller_manifests.json")).unwrap();
    controller_manifests.sort_by(|left, right| left.controller_id.cmp(&right.controller_id));
    let relations = relations();
    let relation_fingerprint = relations.fingerprint().unwrap();
    for bindings in campaign.data_bindings.values_mut() {
        for binding in bindings {
            binding.relation_fingerprint = Some(relation_fingerprint.clone());
        }
    }
    let mut data_identities = campaign
        .data_bindings
        .values()
        .flatten()
        .map(|binding| {
            let mut identity = TrainingDataIdentity {
                requirement_key: data_binding_requirement_key(
                    &binding.node_id,
                    &binding.input_name,
                ),
                schema_fingerprint: binding.schema_fingerprint.clone(),
                plan_fingerprint: binding.plan_fingerprint.clone(),
                relation_fingerprint: relation_fingerprint.clone(),
                data_content_fingerprint: "a".repeat(64),
                target_content_fingerprint: "b".repeat(64),
                identity_fingerprint: "0".repeat(64),
            };
            identity.identity_fingerprint = identity.compute_fingerprint().unwrap();
            identity
        })
        .collect::<Vec<_>>();
    data_identities.sort_by(|left, right| left.requirement_key.cmp(&right.requirement_key));
    let options: TrainingOptions = serde_json::from_value(serde_json::json!({
        "refit": refit,
        "refit_strategy": if refit { serde_json::json!("refit_one") } else { serde_json::Value::Null },
        "seed": 12345,
        "selection": {
            "id": "selection:rmse",
            "metric": {"name": "rmse", "objective": "minimize"},
            "require_finite": true
        },
        "selection_output_id": "output:prediction",
        "outputs": [{
            "output_id": "output:prediction",
            "node_id": "model:base",
            "prediction_level": "sample",
            "unit_level": "physical_sample",
            "prediction_kind": "regression_point",
            "target_names": ["protein"],
            "target_units": ["percent"],
            "class_labels": [[]],
            "output_order": "target_order",
            "target_space": "raw"
        }],
        "scheduler": {"kind": "sequential", "backend": null, "workers": 1},
        "resources": {"cpu_threads": 1, "memory_bytes": null, "gpu_devices": [], "wall_time_ms": null},
        "artifacts": {"cv_artifacts": "discard", "prediction_caches": "retain", "fitted_artifacts": "allow_host_sidecar"}
    }))
    .unwrap();
    let mut request = TrainingRequest {
        schema_version: TRAINING_REQUEST_SCHEMA_VERSION,
        request_id: "training:package.synthetic".to_string(),
        plan_id: "plan:training.package.synthetic".to_string(),
        graph,
        campaign,
        controller_manifests,
        data_identities,
        parameter_patches: Vec::new(),
        patch_policies: Vec::new(),
        influence_requirements: Vec::new(),
        training_losses: Vec::new(),
        options,
        request_fingerprint: "0".repeat(64),
    };
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
    assert!(!stacking);
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
    fixture
        .request
        .controller_manifests
        .sort_by(|left, right| left.controller_id.cmp(&right.controller_id));
    resign_request(&mut fixture.request);
    let projection = fixture.request.project().unwrap();
    fixture.preferred = projection.plan.variants[0].variant_id.clone();
    fixture.influence = influence_manifest(&fixture.request, &projection, &fixture.relations);
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

fn sample(v:&str)->SampleId { SampleId::new(v).unwrap() }
fn node(v:&str)->NodeId { NodeId::new(v).unwrap() }
fn content_hash(v:&str)->String { format!("{:x}",Sha256::digest(v.as_bytes())) }
