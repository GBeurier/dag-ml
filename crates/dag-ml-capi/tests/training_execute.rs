//! End-to-end and ownership tests for the native training C ABI
//! (`dagml_training_execute`).
//!
//! The controllers and data provider are exercised through the real C vtables:
//! a small bridge stores a native `RuntimeController` in the vtable `user_data`
//! and serializes `NodeTask`/`NodeResult` across the boundary, so the training
//! run drives the same proven fixture as the core integration test while going
//! through `dagml_training_execute` for real. Atomic counters on the bridges
//! prove the documented ownership contract: no callback fires before a preflight
//! refusal, each owning `user_data` is destroyed exactly once, and the returned
//! result keeps controller/artifact handles alive until it is freed.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::slice;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use dag_ml_capi::{
    dagml_owned_bytes_free, dagml_string_free, dagml_training_execute, dagml_training_result_free,
    dagml_training_result_outcome_json, dagml_training_result_replay, DagMlBytesView,
    DagMlControllerBinding, DagMlControllerVTable, DagMlDataVTable, DagMlHandle, DagMlOwnedBytes,
    DagMlStatusCode, DagMlString, DagMlTrainingExecuteRequest, DagMlTrainingReplayRequest,
    DagMlTrainingResult, DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION,
    DAG_ML_CONTROLLER_VTABLE_OWNED_ABI_VERSION, DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION,
};
use dag_ml_core::*;

const REQUEST_FIXTURE: &str =
    include_str!("../../../examples/fixtures/training/training_request_refit.v1.json");

// ---------------------------------------------------------------------------
// Native fixture (ported from crates/dag-ml-core/tests/training_runtime_operation.rs)
// ---------------------------------------------------------------------------

#[derive(Default)]
struct CallState {
    fit_counts: Mutex<BTreeMap<VariantId, usize>>,
    next_handle: AtomicU64,
    preferred: Mutex<Option<VariantId>>,
}

impl CallState {
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
    explicit_model_ports: bool,
    fail_invoke: Arc<std::sync::atomic::AtomicBool>,
}

impl RuntimeController for TrainingController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        if self.fail_invoke.load(Ordering::SeqCst) {
            return Err(DagMlError::RuntimeValidation(
                "test controller was configured to fail".to_string(),
            ));
        }
        let is_model = task.node_plan.node_id.as_str() == "model:base";
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
            {
                let mut counts = self.state.fit_counts.lock().unwrap();
                *counts.entry(variant.clone()).or_default() += 1;
            }
            value = if variant == preferred { 0.0 } else { 5.0 };
        }

        let partition = if matches!(task.phase, Phase::Refit | Phase::Predict | Phase::Explain) {
            PredictionPartition::Final
        } else {
            PredictionPartition::Validation
        };
        let explicit_model_ports = is_model && self.explicit_model_ports;
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
                target_names: vec![self.prediction_name.clone()],
            }]
        } else {
            Vec::new()
        };
        if explicit_model_ports {
            let mut sibling = predictions
                .first()
                .expect("explicit C ABI model port controller emits the primary prediction")
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
        let regression_targets = if is_model && task.phase == Phase::FitCv {
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
        let mut outputs = BTreeMap::from([(
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
        )]);
        if explicit_model_ports {
            outputs.insert(
                "probability".to_string(),
                HandleRef {
                    handle: self.state.handle(),
                    kind: HandleKind::Prediction,
                    owner_controller: self.id.clone(),
                },
            );
        }
        Ok(NodeResult {
            schema_version: None,
            node_id: task.node_plan.node_id.clone(),
            outputs,
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            explanations: Vec::new(),
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

fn fixture() -> Fixture {
    let mut request_json: serde_json::Value = serde_json::from_str(REQUEST_FIXTURE).unwrap();
    request_json["options"]["selection_output_id"] =
        serde_json::Value::String("output:prediction".to_string());
    let mut request: TrainingRequest = serde_json::from_value(request_json).unwrap();
    request.options.refit = true;
    request.options.refit_strategy = Some(RefitStrategy::RefitOne);
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
    let all = projection
        .plan
        .fold_set
        .as_ref()
        .unwrap()
        .sample_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let oof_consumers = projection
        .plan
        .graph_plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.contract.requires_oof)
        .map(|edge| edge.target.node_id.clone())
        .collect::<BTreeSet<_>>();
    let mut coordinates = Vec::<(
        TrainingInfluenceKind,
        String,
        Option<NodeId>,
        BTreeSet<SampleId>,
    )>::new();
    for node_id in &projection.predictor_node_ids {
        let plan = &projection.plan.node_plans[node_id];
        if matches!(
            plan.fit_scope,
            ControllerFitScope::Stateless | ControllerFitScope::InferenceOnly
        ) {
            continue;
        }
        let kind = if oof_consumers.contains(node_id)
            || plan
                .controller_capabilities
                .contains(&ControllerCapability::TrainsAggregation)
        {
            TrainingInfluenceKind::TrainedMetaAggregation
        } else if plan.kind == NodeKind::Model {
            TrainingInfluenceKind::ModelFit
        } else if plan.kind == NodeKind::Tuner {
            TrainingInfluenceKind::HpoSelection
        } else {
            TrainingInfluenceKind::TransformFit
        };
        if plan.supported_phases.contains(&Phase::FitCv) {
            for fold in &projection.plan.fold_set.as_ref().unwrap().folds {
                coordinates.push((
                    kind,
                    format!("fit_cv:{}", fold.fold_id),
                    Some(node_id.clone()),
                    fold.train_sample_ids.iter().cloned().collect(),
                ));
            }
        }
        if request.options.refit && plan.supported_phases.contains(&Phase::Refit) {
            coordinates.push((
                kind,
                "refit:full".to_string(),
                Some(node_id.clone()),
                all.clone(),
            ));
        }
    }
    coordinates.push((
        TrainingInfluenceKind::HpoSelection,
        format!("select:{}", request.options.selection.id),
        None,
        all,
    ));
    for requirement in &request.influence_requirements {
        coordinates.push((
            requirement.kind,
            requirement.scope_id.clone(),
            Some(requirement.node_id.clone()),
            requirement.physical_sample_ids.iter().cloned().collect(),
        ));
    }
    let mut entries = coordinates
        .into_iter()
        .map(
            |(kind, scope_id, node_id, samples)| TrainingInfluenceEntry {
                kind,
                scope_id,
                node_id,
                physical_sample_ids: samples.iter().cloned().collect(),
                origin_sample_ids: Vec::new(),
                group_ids: relations
                    .records
                    .iter()
                    .filter(|relation| samples.contains(&relation.sample_id))
                    .filter_map(|relation| relation.group_id.clone())
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
            },
        )
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        (&left.kind, &left.scope_id, &left.node_id).cmp(&(
            &right.kind,
            &right.scope_id,
            &right.node_id,
        ))
    });
    let mut manifest = TrainingInfluenceManifest {
        schema_version: TRAINING_INFLUENCE_MANIFEST_SCHEMA_VERSION,
        relation_fingerprint: relations.fingerprint().unwrap(),
        entries,
        manifest_fingerprint: "0".repeat(64),
    };
    manifest.manifest_fingerprint = manifest.compute_fingerprint().unwrap();
    manifest
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

fn add_explicit_model_probability_port(fixture: &mut Fixture) {
    let mut probability = fixture
        .request
        .graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .outputs
        .iter()
        .find(|port| port.name == "oof")
        .unwrap()
        .clone();
    probability.name = "probability".to_string();
    fixture
        .request
        .graph
        .nodes
        .iter_mut()
        .find(|node| node.id.as_str() == "model:base")
        .unwrap()
        .ports
        .outputs
        .push(probability.clone());
    fixture
        .request
        .controller_manifests
        .iter_mut()
        .find(|manifest| manifest.controller_id.as_str() == "controller:model.mock")
        .unwrap()
        .output_ports
        .push(probability);
    fixture.request.options.outputs[0].port_name = Some("oof".to_string());
    rebuild(fixture);
}

fn sample(value: &str) -> SampleId {
    SampleId::new(value).unwrap()
}

fn node(value: &str) -> NodeId {
    NodeId::new(value).unwrap()
}

fn envelopes_for(fixture: &Fixture) -> BTreeMap<String, ExternalDataPlanEnvelope> {
    let mut envelopes = BTreeMap::new();
    let identities = fixture
        .request
        .data_identities
        .iter()
        .map(|identity| (identity.requirement_key.clone(), identity))
        .collect::<BTreeMap<_, _>>();
    for bindings in fixture.request.campaign.data_bindings.values() {
        for binding in bindings {
            let key = data_binding_requirement_key(&binding.node_id, &binding.input_name);
            let identity = identities.get(&key).expect("identity for binding");
            envelopes.insert(
                key,
                ExternalDataPlanEnvelope {
                    schema_version: EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION,
                    schema_fingerprint: binding.schema_fingerprint.clone(),
                    plan_fingerprint: binding.plan_fingerprint.clone(),
                    relation_fingerprint: binding.relation_fingerprint.clone(),
                    data_content_fingerprint: Some(identity.data_content_fingerprint.clone()),
                    target_content_fingerprint: Some(identity.target_content_fingerprint.clone()),
                    coordinator_relations: Some(fixture.relations.clone()),
                },
            );
        }
    }
    envelopes
}

// ---------------------------------------------------------------------------
// C vtable bridges over the native controller / data provider
// ---------------------------------------------------------------------------

struct ControllerHost {
    controller: Box<dyn RuntimeController>,
    invoked: Arc<AtomicUsize>,
    destroyed: Arc<AtomicUsize>,
    released: Arc<Mutex<Vec<u64>>>,
    result_wire_mutation: ResultWireMutation,
}

#[derive(Clone, Copy, Default)]
enum ResultWireMutation {
    #[default]
    None,
    UnknownRootField,
    PositionalOutputHandle,
}

unsafe extern "C" fn host_invoke(
    user_data: *mut c_void,
    task_json: DagMlBytesView,
    out_result_json: *mut DagMlOwnedBytes,
) -> DagMlStatusCode {
    let host = &*(user_data as *const ControllerHost);
    host.invoked.fetch_add(1, Ordering::SeqCst);
    let bytes = slice::from_raw_parts(task_json.ptr, task_json.len);
    let task: NodeTask = match serde_json::from_slice(bytes) {
        Ok(task) => task,
        Err(_) => return DagMlStatusCode::VALIDATION_ERROR,
    };
    match host.controller.invoke(&task) {
        Ok(result) => {
            let mut result_json = serde_json::to_value(&result).expect("serialize node result");
            match host.result_wire_mutation {
                ResultWireMutation::None => {}
                ResultWireMutation::UnknownRootField => {
                    result_json.as_object_mut().unwrap().insert(
                        "unexpected_contract_field".to_string(),
                        serde_json::json!(true),
                    );
                }
                ResultWireMutation::PositionalOutputHandle => {
                    let handle = result_json["outputs"]
                        .as_object_mut()
                        .unwrap()
                        .values_mut()
                        .next()
                        .unwrap();
                    let positional = {
                        let object = handle.as_object().unwrap();
                        serde_json::json!([
                            object["handle"],
                            object["kind"],
                            object["owner_controller"]
                        ])
                    };
                    *handle = positional;
                }
            }
            let mut data = serde_json::to_vec(&result_json).expect("serialize node result");
            let owned = DagMlOwnedBytes {
                ptr: data.as_mut_ptr(),
                len: data.len(),
                capacity: data.capacity(),
            };
            std::mem::forget(data);
            *out_result_json = owned;
            DagMlStatusCode::OK
        }
        Err(_) => DagMlStatusCode::VALIDATION_ERROR,
    }
}

unsafe extern "C" fn host_release_bytes(_user_data: *mut c_void, bytes: DagMlOwnedBytes) {
    if !bytes.ptr.is_null() {
        drop(Vec::from_raw_parts(bytes.ptr, bytes.len, bytes.capacity));
    }
}

unsafe extern "C" fn host_release(user_data: *mut c_void, handle: DagMlHandle) {
    let host = &*(user_data as *const ControllerHost);
    host.released.lock().unwrap().push(handle);
}

unsafe extern "C" fn host_destroy(user_data: *mut c_void) {
    let host = Box::from_raw(user_data as *mut ControllerHost);
    host.destroyed.fetch_add(1, Ordering::SeqCst);
    drop(host);
}

fn controller_vtable(host: Box<ControllerHost>) -> DagMlControllerVTable {
    DagMlControllerVTable {
        abi_version: DAG_ML_CONTROLLER_VTABLE_OWNED_ABI_VERSION,
        user_data: Box::into_raw(host) as *mut c_void,
        clone_with: None,
        describe: None,
        fit: None,
        predict: None,
        invoke: Some(host_invoke),
        release_bytes: Some(host_release_bytes),
        release: Some(host_release),
        destroy: Some(host_destroy),
    }
}

struct DataHost {
    next_handle: AtomicU64,
    destroyed: Arc<AtomicUsize>,
}

unsafe extern "C" fn data_materialize(
    user_data: *mut c_void,
    _dataset: DagMlHandle,
    _request_json: DagMlBytesView,
    out_handle: *mut DagMlHandle,
) -> DagMlStatusCode {
    let host = &*(user_data as *const DataHost);
    *out_handle = host.next_handle.fetch_add(1, Ordering::SeqCst) + 1;
    DagMlStatusCode::OK
}

unsafe extern "C" fn data_make_view(
    user_data: *mut c_void,
    _data: DagMlHandle,
    _selector_json: DagMlBytesView,
    out_view: *mut DagMlHandle,
) -> DagMlStatusCode {
    let host = &*(user_data as *const DataHost);
    *out_view = host.next_handle.fetch_add(1, Ordering::SeqCst) + 1;
    DagMlStatusCode::OK
}

unsafe extern "C" fn data_release(_user_data: *mut c_void, _handle: DagMlHandle) {}

unsafe extern "C" fn data_destroy(user_data: *mut c_void) {
    let host = Box::from_raw(user_data as *mut DataHost);
    host.destroyed.fetch_add(1, Ordering::SeqCst);
    drop(host);
}

fn data_vtable(host: Box<DataHost>) -> DagMlDataVTable {
    DagMlDataVTable {
        abi_version: DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION,
        user_data: Box::into_raw(host) as *mut c_void,
        materialize: Some(data_materialize),
        make_view: Some(data_make_view),
        view_identity: None,
        target_arrow: None,
        feature_arrow: None,
        release: Some(data_release),
        destroy: Some(data_destroy),
    }
}

fn bytes_view(bytes: &[u8]) -> DagMlBytesView {
    DagMlBytesView {
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    }
}

/// Shared observation counters for one training call.
struct Counters {
    invoked: Arc<AtomicUsize>,
    controller_destroyed: Arc<AtomicUsize>,
    data_destroyed: Arc<AtomicUsize>,
    released: Arc<Mutex<Vec<u64>>>,
}

/// One controller binding plus the owned buffers its `DagMlBytesView`s point at.
struct Binding {
    _id: Box<str>,
    binding: DagMlControllerBinding,
}

fn make_controller(
    id: &str,
    controller: Box<dyn RuntimeController>,
    counters: &Counters,
) -> Binding {
    let host = Box::new(ControllerHost {
        controller,
        invoked: counters.invoked.clone(),
        destroyed: counters.controller_destroyed.clone(),
        released: counters.released.clone(),
        result_wire_mutation: ResultWireMutation::None,
    });
    let id: Box<str> = id.into();
    let binding = DagMlControllerBinding {
        controller_id: bytes_view(id.as_bytes()),
        vtable: controller_vtable(host),
    };
    Binding { _id: id, binding }
}

/// Everything a `dagml_training_execute` call borrows for its duration.
struct Inputs {
    request_json: Vec<u8>,
    relations_json: Vec<u8>,
    influence_json: Vec<u8>,
    envelopes_json: Vec<u8>,
    outcome_id: Box<str>,
    run_id: Box<str>,
    bundle_id: Box<str>,
    data_owner: Box<str>,
    bindings: Vec<Binding>,
    data_provider: DagMlDataVTable,
}

impl Inputs {
    fn request(&self) -> DagMlTrainingExecuteRequest {
        let raw_bindings = self
            .bindings
            .iter()
            .map(|binding| binding.binding)
            .collect::<Vec<_>>();
        // Leak the flattened binding array for the (synchronous) call duration;
        // reclaimed by the test process exit. The pointer stays valid because
        // `dagml_training_execute` returns before `request()`'s result is used.
        let boxed = raw_bindings.into_boxed_slice();
        let count = boxed.len();
        let ptr = Box::into_raw(boxed) as *const DagMlControllerBinding;
        DagMlTrainingExecuteRequest {
            request_json: bytes_view(&self.request_json),
            outcome_id: bytes_view(self.outcome_id.as_bytes()),
            run_id: bytes_view(self.run_id.as_bytes()),
            bundle_id: bytes_view(self.bundle_id.as_bytes()),
            relations_json: bytes_view(&self.relations_json),
            influence_json: bytes_view(&self.influence_json),
            envelopes_json: bytes_view(&self.envelopes_json),
            warnings_json: DagMlBytesView {
                ptr: std::ptr::null(),
                len: 0,
            },
            diagnostics_json: DagMlBytesView {
                ptr: std::ptr::null(),
                len: 0,
            },
            dataset: 1,
            data_provider: self.data_provider,
            data_owner_controller_id: bytes_view(self.data_owner.as_bytes()),
            controller_bindings: ptr,
            controller_binding_count: count,
        }
    }
}

/// Build the standard two-controller success setup with fresh counters.
fn build_inputs(counters: &Counters, model_fails: bool) -> (Fixture, Inputs) {
    build_inputs_with(counters, model_fails, false, |_| {})
}

fn build_inputs_with(
    counters: &Counters,
    model_fails: bool,
    explicit_model_ports: bool,
    mutate_fixture: impl FnOnce(&mut Fixture),
) -> (Fixture, Inputs) {
    let mut fixture = fixture();
    mutate_fixture(&mut fixture);
    let envelopes = envelopes_for(&fixture);
    let state = Arc::new(CallState::default());
    *state.preferred.lock().unwrap() = Some(fixture.preferred.clone());
    let never = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let model_flag = Arc::new(std::sync::atomic::AtomicBool::new(model_fails));

    let transform = make_controller(
        "controller:transform.mock",
        Box::new(TrainingController {
            id: ControllerId::new("controller:transform.mock").unwrap(),
            state: state.clone(),
            emits_predictions: false,
            emits_artifact: false,
            prediction_name: "aux".to_string(),
            explicit_model_ports: false,
            fail_invoke: never,
        }),
        counters,
    );
    let model = make_controller(
        "controller:model.mock",
        Box::new(TrainingController {
            id: ControllerId::new("controller:model.mock").unwrap(),
            state,
            emits_predictions: true,
            emits_artifact: true,
            prediction_name: "protein".to_string(),
            explicit_model_ports,
            fail_invoke: model_flag,
        }),
        counters,
    );

    let data_provider = data_vtable(Box::new(DataHost {
        next_handle: AtomicU64::new(0),
        destroyed: counters.data_destroyed.clone(),
    }));

    let inputs = Inputs {
        request_json: serde_json::to_vec(&fixture.request).unwrap(),
        relations_json: serde_json::to_vec(&fixture.relations).unwrap(),
        influence_json: serde_json::to_vec(&fixture.influence).unwrap(),
        envelopes_json: serde_json::to_vec(&envelopes).unwrap(),
        outcome_id: "outcome:c.native".into(),
        run_id: "run:c.native".into(),
        bundle_id: "bundle:c.native".into(),
        data_owner: "controller:model.mock".into(),
        bindings: vec![transform, model],
        data_provider,
    };
    (fixture, inputs)
}

fn fresh_counters() -> Counters {
    Counters {
        invoked: Arc::new(AtomicUsize::new(0)),
        controller_destroyed: Arc::new(AtomicUsize::new(0)),
        data_destroyed: Arc::new(AtomicUsize::new(0)),
        released: Arc::new(Mutex::new(Vec::new())),
    }
}

unsafe fn error_text(error: &DagMlString) -> String {
    if error.ptr.is_null() {
        return String::new();
    }
    let bytes = slice::from_raw_parts(error.ptr as *const u8, error.len);
    String::from_utf8_lossy(bytes).into_owned()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn training_execute_end_to_end_success_keeps_handles_alive_until_free() {
    let counters = fresh_counters();
    let (_fixture, inputs) = build_inputs(&counters, false);
    let request = inputs.request();
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();

    let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
    assert_eq!(
        status,
        DagMlStatusCode::OK,
        "unexpected failure: {}",
        unsafe { error_text(&error) }
    );
    assert!(!result.is_null());
    assert!(counters.invoked.load(Ordering::SeqCst) > 0);
    // The result owns the registry, so nothing is destroyed while it is alive.
    assert_eq!(counters.controller_destroyed.load(Ordering::SeqCst), 0);
    assert!(counters.released.lock().unwrap().is_empty());

    // The outcome JSON is independently Rust-validable.
    let mut outcome_json = DagMlOwnedBytes::default();
    let mut getter_error = DagMlString::default();
    let getter_status =
        unsafe { dagml_training_result_outcome_json(result, &mut outcome_json, &mut getter_error) };
    assert_eq!(getter_status, DagMlStatusCode::OK);
    let json = unsafe { slice::from_raw_parts(outcome_json.ptr, outcome_json.len) }.to_vec();
    let text = String::from_utf8(json).unwrap();
    let outcome = TrainingOutcome::from_json(&text).expect("outcome round-trips through from_json");
    outcome.validate().unwrap();
    assert_eq!(outcome.outcome_id, "outcome:c.native");
    assert_eq!(outcome.refit.status, TrainingRefitStatus::Completed);
    assert_eq!(outcome.replayable_phases, vec![Phase::Predict]);
    assert_eq!(outcome.execution_bundle.refit_artifacts.len(), 1);
    unsafe { dagml_owned_bytes_free(outcome_json) };

    // Freeing releases the model controller's tracked handles and then destroys
    // both owned controller user_data exactly once.
    unsafe { dagml_training_result_free(result) };
    assert_eq!(counters.controller_destroyed.load(Ordering::SeqCst), 2);
    assert!(
        !counters.released.lock().unwrap().is_empty(),
        "model/refit handles are released when the result is freed"
    );
    // The data vtable is borrowed: the call never destroys its user_data. Clean
    // it up here as a well-behaved host would.
    assert_eq!(
        counters.data_destroyed.load(Ordering::SeqCst),
        0,
        "borrowed data vtable user_data is never destroyed by the call"
    );
    unsafe { data_destroy(request.data_provider.user_data) };
}

#[test]
fn training_execute_filters_explicit_multi_prediction_port_outputs() {
    let counters = fresh_counters();
    let (_fixture, inputs) =
        build_inputs_with(&counters, false, true, add_explicit_model_probability_port);
    let request = inputs.request();
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();

    let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
    assert_eq!(
        status,
        DagMlStatusCode::OK,
        "unexpected failure: {}",
        unsafe { error_text(&error) }
    );
    assert!(!result.is_null());

    let mut outcome_json = DagMlOwnedBytes::default();
    let mut getter_error = DagMlString::default();
    let getter_status =
        unsafe { dagml_training_result_outcome_json(result, &mut outcome_json, &mut getter_error) };
    assert_eq!(getter_status, DagMlStatusCode::OK);
    let json = unsafe { slice::from_raw_parts(outcome_json.ptr, outcome_json.len) }.to_vec();
    let text = String::from_utf8(json).unwrap();
    let outcome = TrainingOutcome::from_json(&text).expect("multi-port outcome validates");
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
    assert!(outcome.score_set.reports.iter().any(|report| {
        report.producer_node.as_str() == "model:base"
            && report.producer_port.as_deref() == Some("probability")
    }));
    unsafe {
        dagml_owned_bytes_free(outcome_json);
        dagml_training_result_free(result);
        data_destroy(request.data_provider.user_data);
    }
}

#[test]
fn training_result_replay_predict_over_c_abi_uses_live_attached_result() {
    let counters = fresh_counters();
    let (_fixture, inputs) =
        build_inputs_with(&counters, false, true, add_explicit_model_probability_port);
    let request = inputs.request();
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();

    let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
    assert_eq!(
        status,
        DagMlStatusCode::OK,
        "unexpected training failure: {}",
        unsafe { error_text(&error) }
    );
    assert!(!result.is_null());

    let mut outcome_json = DagMlOwnedBytes::default();
    let mut getter_error = DagMlString::default();
    let getter_status =
        unsafe { dagml_training_result_outcome_json(result, &mut outcome_json, &mut getter_error) };
    assert_eq!(getter_status, DagMlStatusCode::OK);
    let source_text = String::from_utf8(
        unsafe { slice::from_raw_parts(outcome_json.ptr, outcome_json.len) }.to_vec(),
    )
    .unwrap();
    let source = TrainingOutcome::from_json(&source_text).expect("source outcome validates");
    unsafe { dagml_owned_bytes_free(outcome_json) };

    let mut replay_request = TrainingReplayRequest {
        schema_version: TRAINING_REPLAY_REQUEST_SCHEMA_VERSION,
        request_id: "replay:c.native.predict".to_string(),
        source_outcome_fingerprint: source.outcome_fingerprint.clone(),
        phase: Phase::Predict,
        data_envelope_keys: serde_json::from_slice::<BTreeMap<String, ExternalDataPlanEnvelope>>(
            &inputs.envelopes_json,
        )
        .unwrap()
        .keys()
        .cloned()
        .collect(),
        output_binding_ids: source
            .outputs
            .iter()
            .map(|output| output.binding.binding_id.clone())
            .collect(),
        request_fingerprint: "0".repeat(64),
    };
    replay_request.request_fingerprint = replay_request.compute_fingerprint().unwrap();
    let replay_request_json = serde_json::to_vec(&replay_request).unwrap();
    let replay_outcome_id: Box<str> = "outcome:c.native.replay.predict".into();
    let replay_run_id: Box<str> = "run:c.native.replay.predict".into();
    let replay_call = DagMlTrainingReplayRequest {
        replay_request_json: bytes_view(&replay_request_json),
        outcome_id: bytes_view(replay_outcome_id.as_bytes()),
        run_id: bytes_view(replay_run_id.as_bytes()),
        data_envelopes_json: bytes_view(&inputs.envelopes_json),
        warnings_json: DagMlBytesView {
            ptr: std::ptr::null(),
            len: 0,
        },
        diagnostics_json: DagMlBytesView {
            ptr: std::ptr::null(),
            len: 0,
        },
        dataset: 1,
        data_provider: request.data_provider,
        data_owner_controller_id: bytes_view(inputs.data_owner.as_bytes()),
    };

    let mut replay_json = DagMlOwnedBytes::default();
    let mut replay_error = DagMlString::default();
    let replay_status = unsafe {
        dagml_training_result_replay(result, &replay_call, &mut replay_json, &mut replay_error)
    };
    assert_eq!(
        replay_status,
        DagMlStatusCode::OK,
        "unexpected replay failure: {}",
        unsafe { error_text(&replay_error) }
    );
    let replay_text = String::from_utf8(
        unsafe { slice::from_raw_parts(replay_json.ptr, replay_json.len) }.to_vec(),
    )
    .unwrap();
    let replay_outcome =
        TrainingReplayOutcome::from_json(&replay_text).expect("replay outcome validates");
    replay_outcome
        .validate_against(&source, &replay_request)
        .expect("replay cross-links source and request");
    assert_eq!(replay_outcome.phase, Phase::Predict);
    assert_eq!(replay_outcome.outputs.len(), 1);
    let output = &replay_outcome.outputs[0];
    assert_eq!(output.binding.port_name, "oof");
    assert!(output.predictions.iter().all(|block| {
        block.producer_node.as_str() == "model:base"
            && block.producer_port.as_deref() == Some("oof")
            && block.partition == PredictionPartition::Final
            && block.fold_id.is_none()
    }));
    assert!(replay_outcome.explanations.is_empty());

    unsafe {
        dagml_owned_bytes_free(replay_json);
        dagml_training_result_free(result);
        data_destroy(request.data_provider.user_data);
    }
}

#[test]
fn training_result_free_and_getter_handle_null() {
    // Freeing null is a no-op; the getter rejects null without writing bytes.
    unsafe { dagml_training_result_free(std::ptr::null_mut()) };
    let mut out = DagMlOwnedBytes::default();
    let mut error = DagMlString::default();
    let status =
        unsafe { dagml_training_result_outcome_json(std::ptr::null(), &mut out, &mut error) };
    assert_eq!(status, DagMlStatusCode::INVALID_ARGUMENT);
    assert!(out.ptr.is_null());
}

#[test]
fn training_execute_rejects_null_arguments() {
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();
    // Null request struct.
    let status = unsafe { dagml_training_execute(std::ptr::null(), &mut result, &mut error) };
    assert_eq!(status, DagMlStatusCode::INVALID_ARGUMENT);
    assert!(result.is_null());
    unsafe { dagml_string_free(error) };
    error = DagMlString::default();

    // Null out_result.
    let counters = fresh_counters();
    let (_fixture, inputs) = build_inputs(&counters, false);
    let request = inputs.request();
    let status = unsafe { dagml_training_execute(&request, std::ptr::null_mut(), &mut error) };
    assert_eq!(status, DagMlStatusCode::INVALID_ARGUMENT);
    // No controller was consumed because ownership transfer had not begun.
    assert_eq!(counters.invoked.load(Ordering::SeqCst), 0);
    assert_eq!(counters.controller_destroyed.load(Ordering::SeqCst), 0);
    unsafe {
        dagml_string_free(error);
        for binding in &inputs.bindings {
            host_destroy(binding.binding.vtable.user_data);
        }
        data_destroy(request.data_provider.user_data);
    }
    assert_eq!(counters.controller_destroyed.load(Ordering::SeqCst), 2);
}

#[test]
fn training_execute_duplicate_envelope_key_refused_before_callbacks() {
    let counters = fresh_counters();
    let (fixture, mut inputs) = build_inputs(&counters, false);
    let key = data_binding_requirement_key(&node("model:base"), "x");
    let envelope = serde_json::to_string(&envelopes_for(&fixture)[&key]).unwrap();
    // A hand-built object with the SAME key twice: serde would silently keep the
    // last, but the strict TCV1 pre-pass must reject it before any BTreeMap.
    inputs.envelopes_json = format!("{{\"{key}\":{envelope},\"{key}\":{envelope}}}").into_bytes();
    let request = inputs.request();
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();

    let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
    assert!(result.is_null());
    assert_eq!(
        counters.invoked.load(Ordering::SeqCst),
        0,
        "duplicate key is rejected before any controller callback"
    );
    // Controllers were taken and cleaned up exactly once each.
    assert_eq!(counters.controller_destroyed.load(Ordering::SeqCst), 2);
}

#[test]
fn training_execute_rejects_closed_relation_fields_before_callbacks() {
    for mutation in [
        (|inputs: &mut Inputs| {
            let mut relations: serde_json::Value =
                serde_json::from_slice(&inputs.relations_json).unwrap();
            relations.as_object_mut().unwrap().insert(
                "unexpected_contract_field".to_string(),
                serde_json::json!(true),
            );
            inputs.relations_json = serde_json::to_vec(&relations).unwrap();
        }) as fn(&mut Inputs),
        |inputs: &mut Inputs| {
            let mut envelopes: serde_json::Value =
                serde_json::from_slice(&inputs.envelopes_json).unwrap();
            envelopes
                .as_object_mut()
                .unwrap()
                .values_mut()
                .next()
                .unwrap()["coordinator_relations"]["records"][0]
                .as_object_mut()
                .unwrap()
                .insert(
                    "unexpected_contract_field".to_string(),
                    serde_json::json!(true),
                );
            inputs.envelopes_json = serde_json::to_vec(&envelopes).unwrap();
        },
    ] {
        let counters = fresh_counters();
        let (_fixture, mut inputs) = build_inputs(&counters, false);
        mutation(&mut inputs);
        let request = inputs.request();
        let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
        let mut error = DagMlString::default();

        let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
        let message = unsafe { error_text(&error) };
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR, "{message}");
        assert!(result.is_null());
        assert!(message.contains("unexpected_contract_field"), "{message}");
        assert_eq!(
            counters.invoked.load(Ordering::SeqCst),
            0,
            "closed relation fields must be refused before callbacks"
        );
        assert_eq!(counters.controller_destroyed.load(Ordering::SeqCst), 2);
        unsafe {
            dagml_string_free(error);
            data_destroy(request.data_provider.user_data);
        }
    }
}

#[test]
fn training_execute_rejects_non_object_or_open_controller_results() {
    for (mutation, expected_error) in [
        (
            ResultWireMutation::UnknownRootField,
            "unexpected_contract_field",
        ),
        (
            ResultWireMutation::PositionalOutputHandle,
            "must use a JSON object at the external contract boundary",
        ),
    ] {
        let counters = fresh_counters();
        let (_fixture, inputs) = build_inputs(&counters, false);
        let host = inputs.bindings[0].binding.vtable.user_data as *mut ControllerHost;
        unsafe { (*host).result_wire_mutation = mutation };
        let request = inputs.request();
        let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
        let mut error = DagMlString::default();

        let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
        let message = unsafe { error_text(&error) };
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR, "{message}");
        assert!(result.is_null());
        assert!(message.contains(expected_error), "{message}");
        assert!(
            counters.invoked.load(Ordering::SeqCst) > 0,
            "the invalid controller result must be observed at the callback boundary"
        );
        assert_eq!(
            counters.controller_destroyed.load(Ordering::SeqCst),
            2,
            "both owning controller hosts are cleaned up after callback refusal"
        );
        unsafe {
            dagml_string_free(error);
            data_destroy(request.data_provider.user_data);
        }
    }
}

#[test]
fn training_execute_requires_release_callback_before_invocation() {
    let counters = fresh_counters();
    let (_fixture, mut inputs) = build_inputs(&counters, false);
    inputs.bindings[0].binding.vtable.release = None;
    let request = inputs.request();
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();

    let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
    let message = unsafe { error_text(&error) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR, "{message}");
    assert!(result.is_null());
    assert!(message.contains("missing release callback"), "{message}");
    assert_eq!(
        counters.invoked.load(Ordering::SeqCst),
        0,
        "training vtable ownership is rejected before a callback can create handles"
    );
    assert_eq!(
        counters.controller_destroyed.load(Ordering::SeqCst),
        2,
        "a missing handle releaser must not leak controller owners"
    );
    unsafe {
        dagml_string_free(error);
        data_destroy(request.data_provider.user_data);
    }
}

#[test]
fn training_execute_missing_and_extra_envelope_refused_before_callbacks() {
    for mutate in [
        // Empty map: the model:base.x envelope is missing.
        (|json: &mut Vec<u8>| *json = b"{}".to_vec()) as fn(&mut Vec<u8>),
        // Extra, unexpected key breaks exact coverage.
        |json: &mut Vec<u8>| {
            let text = String::from_utf8(json.clone()).unwrap();
            let trimmed = text.trim_end().trim_end_matches('}');
            *json = format!("{trimmed},\"model:base.extra\":{{}}}}").into_bytes();
        },
    ] {
        let counters = fresh_counters();
        let (_fixture, mut inputs) = build_inputs(&counters, false);
        mutate(&mut inputs.envelopes_json);
        let request = inputs.request();
        let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
        let mut error = DagMlString::default();

        let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
        assert!(result.is_null());
        assert_eq!(
            counters.invoked.load(Ordering::SeqCst),
            0,
            "envelope coverage mismatch is refused before any callback"
        );
        assert_eq!(counters.controller_destroyed.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn training_execute_shared_owning_user_data_refused_without_double_destroy() {
    let counters = fresh_counters();
    let (_fixture, inputs) = build_inputs(&counters, false);

    // Two bindings that point at the SAME owning user_data/destroy. This would
    // double-destroy on cleanup, so it must be refused — while still destroying
    // the shared pointer exactly once.
    let host = Box::new(ControllerHost {
        controller: Box::new(TrainingController {
            id: ControllerId::new("controller:model.mock").unwrap(),
            state: Arc::new(CallState::default()),
            emits_predictions: true,
            emits_artifact: true,
            prediction_name: "protein".to_string(),
            explicit_model_ports: false,
            fail_invoke: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }),
        invoked: counters.invoked.clone(),
        destroyed: counters.controller_destroyed.clone(),
        released: counters.released.clone(),
        result_wire_mutation: ResultWireMutation::None,
    });
    let vtable = controller_vtable(host);
    let first_id: Box<str> = "controller:transform.mock".into();
    let second_id: Box<str> = "controller:model.mock".into();
    let shared = [
        DagMlControllerBinding {
            controller_id: bytes_view(first_id.as_bytes()),
            vtable,
        },
        DagMlControllerBinding {
            controller_id: bytes_view(second_id.as_bytes()),
            vtable,
        },
    ];

    let mut request = inputs.request();
    request.controller_bindings = shared.as_ptr();
    request.controller_binding_count = shared.len();
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();

    let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
    assert!(result.is_null());
    assert_eq!(counters.invoked.load(Ordering::SeqCst), 0);
    assert_eq!(
        counters.controller_destroyed.load(Ordering::SeqCst),
        1,
        "the shared owning user_data is destroyed exactly once, never twice"
    );
    assert!(unsafe { error_text(&error) }.contains("double-destroy"));
    // `inputs`'s own two controllers were never handed to this call; drop them
    // so their owned user_data does not leak the test process.
    for binding in inputs.bindings {
        unsafe { host_destroy(binding.binding.vtable.user_data) };
    }
    unsafe { data_destroy(inputs.data_provider.user_data) };
}

#[test]
fn training_execute_controller_failure_cleans_up_exactly_once() {
    let counters = fresh_counters();
    let (_fixture, inputs) = build_inputs(&counters, true);
    let request = inputs.request();
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();

    let status = unsafe { dagml_training_execute(&request, &mut result, &mut error) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
    assert!(result.is_null());
    // The failing controller callback fired, then every owning controller
    // user_data was destroyed exactly once during cleanup.
    assert!(counters.invoked.load(Ordering::SeqCst) > 0);
    assert_eq!(counters.controller_destroyed.load(Ordering::SeqCst), 2);
}

// ---------------------------------------------------------------------------
// Focused controller-ownership escrow tests (pre-wrapper failure paths)
// ---------------------------------------------------------------------------

struct NoopController {
    id: ControllerId,
}

impl RuntimeController for NoopController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, _task: &NodeTask) -> Result<NodeResult> {
        Err(DagMlError::RuntimeValidation("noop controller".to_string()))
    }
}

/// An owned-ABI controller vtable whose destroy/invoke/release feed `counters`.
/// `with_invoke == false` yields an otherwise-owning vtable that
/// `CAbiRuntimeController::new` must reject (missing `invoke`).
fn owning_controller_vtable(counters: &Counters, with_invoke: bool) -> DagMlControllerVTable {
    let host = Box::new(ControllerHost {
        controller: Box::new(NoopController {
            id: ControllerId::new("controller:noop").unwrap(),
        }),
        invoked: counters.invoked.clone(),
        destroyed: counters.controller_destroyed.clone(),
        released: counters.released.clone(),
        result_wire_mutation: ResultWireMutation::None,
    });
    let mut vtable = controller_vtable(host);
    if !with_invoke {
        vtable.invoke = None;
    }
    vtable
}

/// Drive `dagml_training_execute` for its controller-registry preflight only.
/// The registry builder runs before any JSON is parsed, so the borrowed data
/// vtable is never built; it is cleaned up here (the call never touches it).
unsafe fn execute_controller_bindings(
    bindings: &[DagMlControllerBinding],
) -> (DagMlStatusCode, *mut DagMlTrainingResult, DagMlString) {
    let data = data_vtable(Box::new(DataHost {
        next_handle: AtomicU64::new(0),
        destroyed: Arc::new(AtomicUsize::new(0)),
    }));
    let null_view = DagMlBytesView {
        ptr: std::ptr::null(),
        len: 0,
    };
    let request = DagMlTrainingExecuteRequest {
        request_json: bytes_view(b"{}"),
        outcome_id: bytes_view(b"outcome:x"),
        run_id: bytes_view(b"run:x"),
        bundle_id: bytes_view(b"bundle:x"),
        relations_json: bytes_view(b"{}"),
        influence_json: bytes_view(b"{}"),
        envelopes_json: bytes_view(b"{}"),
        warnings_json: null_view,
        diagnostics_json: null_view,
        dataset: 1,
        data_provider: data,
        data_owner_controller_id: bytes_view(b"controller:model.mock"),
        controller_bindings: bindings.as_ptr(),
        controller_binding_count: bindings.len(),
    };
    let mut result: *mut DagMlTrainingResult = std::ptr::null_mut();
    let mut error = DagMlString::default();
    let status = dagml_training_execute(&request, &mut result, &mut error);
    data_destroy(data.user_data);
    (status, result, error)
}

#[test]
fn training_execute_invalid_controller_id_destroys_owner_once() {
    let counters = fresh_counters();
    // A non-UTF-8 controller id fails in `parse_controller_id_view`, before any
    // wrapper is built for this owning binding.
    let binding = DagMlControllerBinding {
        controller_id: bytes_view(&[0xff, 0xfe]),
        vtable: owning_controller_vtable(&counters, true),
    };
    let (status, result, _error) = unsafe { execute_controller_bindings(&[binding]) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
    assert!(result.is_null());
    assert_eq!(counters.invoked.load(Ordering::SeqCst), 0);
    assert_eq!(
        counters.controller_destroyed.load(Ordering::SeqCst),
        1,
        "the owning user_data is destroyed exactly once even though its wrapper was never built"
    );
}

#[test]
fn training_execute_invalid_vtable_destroys_owner_once() {
    let counters = fresh_counters();
    // Owned ABI + non-null destroy but missing `invoke`: rejected by
    // `CAbiRuntimeController::new`, again before a wrapper exists.
    let binding = DagMlControllerBinding {
        controller_id: bytes_view(b"controller:model.mock"),
        vtable: owning_controller_vtable(&counters, false),
    };
    let (status, result, _error) = unsafe { execute_controller_bindings(&[binding]) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
    assert!(result.is_null());
    assert_eq!(counters.invoked.load(Ordering::SeqCst), 0);
    assert_eq!(
        counters.controller_destroyed.load(Ordering::SeqCst),
        1,
        "an invalid owning vtable is still destroyed exactly once"
    );
}

#[test]
fn training_execute_later_invalid_binding_destroys_every_owner_once() {
    let counters = fresh_counters();
    // A valid owning binding followed by an invalid owning binding: the first
    // wrapper is built (and disarmed), the second is rejected. On the failed
    // return the first is destroyed by the dropped registry and the second by
    // the escrow — each exactly once, with no callback ever invoked.
    let bindings = [
        DagMlControllerBinding {
            controller_id: bytes_view(b"controller:a"),
            vtable: owning_controller_vtable(&counters, true),
        },
        DagMlControllerBinding {
            controller_id: bytes_view(b"controller:b"),
            vtable: owning_controller_vtable(&counters, false),
        },
    ];
    let (status, result, _error) = unsafe { execute_controller_bindings(&bindings) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
    assert!(result.is_null());
    assert_eq!(counters.invoked.load(Ordering::SeqCst), 0);
    assert_eq!(
        counters.controller_destroyed.load(Ordering::SeqCst),
        2,
        "both distinct owning user_data are destroyed exactly once"
    );
}

#[test]
fn training_execute_shared_pointer_with_distinct_owner_destroys_each_once() {
    let counters = fresh_counters();
    // Two bindings share one owning user_data (a double-destroy hazard) and a
    // third advertises a distinct owner. The shared configuration is refused
    // before any wrapper/callback, yet each DISTINCT owner is destroyed once.
    let shared = owning_controller_vtable(&counters, true);
    let distinct = owning_controller_vtable(&counters, true);
    let bindings = [
        DagMlControllerBinding {
            controller_id: bytes_view(b"controller:a"),
            vtable: shared,
        },
        DagMlControllerBinding {
            controller_id: bytes_view(b"controller:b"),
            vtable: shared,
        },
        DagMlControllerBinding {
            controller_id: bytes_view(b"controller:c"),
            vtable: distinct,
        },
    ];
    let (status, result, error) = unsafe { execute_controller_bindings(&bindings) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
    assert!(result.is_null());
    assert_eq!(counters.invoked.load(Ordering::SeqCst), 0);
    assert_eq!(
        counters.controller_destroyed.load(Ordering::SeqCst),
        2,
        "the shared owner and the distinct owner are each destroyed exactly once"
    );
    assert!(unsafe { error_text(&error) }.contains("double-destroy"));
}

#[test]
fn training_execute_rejects_borrowed_owned_user_data_alias_in_both_orders() {
    for owned_first in [false, true] {
        let counters = fresh_counters();
        let owned = owning_controller_vtable(&counters, true);
        let mut borrowed = owned;
        borrowed.abi_version = DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION;
        borrowed.destroy = None;
        let (first, second) = if owned_first {
            (owned, borrowed)
        } else {
            (borrowed, owned)
        };
        let bindings = [
            DagMlControllerBinding {
                controller_id: bytes_view(b"controller:a"),
                vtable: first,
            },
            DagMlControllerBinding {
                controller_id: bytes_view(b"controller:b"),
                vtable: second,
            },
        ];

        let (status, result, error) = unsafe { execute_controller_bindings(&bindings) };
        let message = unsafe { error_text(&error) };
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR, "{message}");
        assert!(result.is_null());
        assert_eq!(counters.invoked.load(Ordering::SeqCst), 0);
        assert_eq!(
            counters.controller_destroyed.load(Ordering::SeqCst),
            1,
            "the single owner is destroyed exactly once for owned_first={owned_first}"
        );
        assert!(message.contains("use-after-free"), "{message}");
        unsafe { dagml_string_free(error) };
    }
}
