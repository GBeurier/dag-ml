//! Shared owner propagation with genuine saved roots and measured callback
//! traps. Mock callbacks are refusal-timing controls, not numerical execution.
use dag_ml_core::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

const SOURCE: &str =
    include_str!("fixtures/artifact_facets_b1/outputs/training_outcome.before.json");
const PACKAGE: &str = include_str!("fixtures/artifact_facets_b1/outputs/ppc.before.json");
const REQUEST: &str = include_str!("fixtures/artifact_facets_b1/outputs/replay_request.json");
const ENVELOPES: &str = include_str!("fixtures/artifact_facets_b1/outputs/replay_envelope.json");

#[derive(Default)]
struct Counts {
    model: AtomicUsize,
    hydrate: AtomicUsize,
    data: AtomicUsize,
    store: AtomicUsize,
}
impl Counts {
    fn assert_zero(&self) {
        assert_eq!(self.model.load(Ordering::SeqCst), 0);
        assert_eq!(self.hydrate.load(Ordering::SeqCst), 0);
        assert_eq!(self.data.load(Ordering::SeqCst), 0);
        assert_eq!(self.store.load(Ordering::SeqCst), 0);
    }
}
fn callback_error() -> DagMlError {
    DagMlError::RuntimeValidation("unexpected refusal-test callback".into())
}
struct Controller {
    id: ControllerId,
    counts: Arc<Counts>,
}
impl RuntimeController for Controller {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }
    fn invoke(&self, _: &NodeTask) -> Result<NodeResult> {
        self.counts.model.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
    fn hydrate_artifact_payload(
        &self,
        _: &ArtifactMaterializationRequest,
        _: &[u8],
    ) -> Result<HandleRef> {
        self.counts.hydrate.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
    fn retain_run_resources(
        &self,
        _: &RunId,
    ) -> Result<Option<Arc<dyn std::fmt::Debug + Send + Sync>>> {
        self.counts.model.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
}
struct Provider(Arc<Counts>);
impl RuntimeDataProvider for Provider {
    fn materialize(&self, _: &DataMaterializationRequest) -> Result<HandleRef> {
        self.0.data.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
    fn make_view(&self, _: &DataViewRequest) -> Result<HandleRef> {
        self.0.data.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
    fn coordinator_relations(&self, _: &DataBinding) -> Result<Option<SampleRelationSet>> {
        self.0.data.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
    fn predict_cohort(&self, _: &DataBinding, _: Phase) -> Result<Option<PredictCohort>> {
        self.0.data.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
    fn training_data_identity(&self, _: &DataBinding) -> Result<Option<TrainingDataIdentity>> {
        self.0.data.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
}
struct Store(Arc<Counts>);
impl RuntimeArtifactStore for Store {
    fn materialize(&self, _: &ArtifactMaterializationRequest) -> Result<HandleRef> {
        self.0.store.fetch_add(1, Ordering::SeqCst);
        Err(callback_error())
    }
}
fn controllers(plan: &ExecutionPlan, counts: &Arc<Counts>) -> RuntimeControllerRegistry {
    let mut registry = RuntimeControllerRegistry::new();
    for id in plan
        .node_plans
        .values()
        .map(|node| node.controller_id.clone())
        .collect::<BTreeSet<_>>()
    {
        registry
            .register(Box::new(Controller {
                id,
                counts: counts.clone(),
            }))
            .unwrap();
    }
    registry
}
fn parent() -> (
    TrainingOutcome,
    PortablePredictorPackage,
    TrainingReplayRequest,
    BTreeMap<String, ExternalDataPlanEnvelope>,
) {
    (
        TrainingOutcome::from_json(SOURCE).unwrap(),
        PortablePredictorPackage::from_json(PACKAGE).unwrap(),
        TrainingReplayRequest::from_json(REQUEST).unwrap(),
        serde_json::from_str(ENVELOPES).unwrap(),
    )
}
fn contradict(envelopes: &mut BTreeMap<String, ExternalDataPlanEnvelope>) {
    for envelope in envelopes.values_mut() {
        let cohort = envelope.predict_cohort.as_mut().unwrap();
        cohort.target_names = vec!["contradictory-target".into()];
        cohort.cohort_fingerprint = cohort.fingerprint().unwrap();
        envelope.validate().unwrap();
    }
}
fn child() -> (
    PortableRefitPackageV3,
    TrainingReplayRequest,
    BTreeMap<String, ExternalDataPlanEnvelope>,
) {
    let child = PortableRefitPackageV3::from_json(include_str!(
        "fixtures/artifact_facets/native_role/child_refit_v3.json"
    ))
    .unwrap();
    let relations: SampleRelationSet = serde_json::from_str(include_str!(
        "fixtures/artifact_facets/native_role/target_relations.json"
    ))
    .unwrap();
    let prototype: BTreeMap<String, ExternalDataPlanEnvelope> =
        serde_json::from_str(ENVELOPES).unwrap();
    // Metadata-only current-cohort control for genuine child authority. This
    // does not claim IO projection/value parity with the parent's native data.
    let cohort = prototype.values().next().unwrap().predict_cohort.clone();
    let bundle = child.outcome.to_runtime_replay_bundle().unwrap();
    let envelopes: BTreeMap<_, _> = bundle
        .data_requirements
        .iter()
        .map(|data| {
            (
                data.key(),
                ExternalDataPlanEnvelope {
                    schema_version: 2,
                    schema_fingerprint: data.schema_fingerprint.clone(),
                    plan_fingerprint: data.plan_fingerprint.clone(),
                    relation_fingerprint: Some(relations.fingerprint().unwrap()),
                    coordinator_relations: Some(relations.clone()),
                    data_content_fingerprint: prototype
                        .values()
                        .next()
                        .unwrap()
                        .data_content_fingerprint
                        .clone(),
                    target_content_fingerprint: prototype
                        .values()
                        .next()
                        .unwrap()
                        .target_content_fingerprint
                        .clone(),
                    predict_cohort: cohort.clone(),
                },
            )
        })
        .collect();
    let mut request = TrainingReplayRequest::from_json(REQUEST).unwrap();
    request.source_outcome_fingerprint = child.outcome.outcome_fingerprint.clone();
    request.data_envelope_keys = envelopes.keys().cloned().collect();
    request.output_binding_ids = child
        .outcome
        .output_bindings
        .iter()
        .map(|binding| binding.binding_id.clone())
        .collect();
    request.output_binding_ids.sort();
    request.request_fingerprint = request.compute_fingerprint().unwrap();
    (child, request, envelopes)
}

#[test]
fn genuine_public_preflights_refuse_explicit_target_contradictions() {
    let (_, package, request, mut envelopes) = parent();
    preflight_loaded_predictor_replay(&package, &request, &envelopes).unwrap();
    contradict(&mut envelopes);
    assert!(matches!(
        preflight_loaded_predictor_replay(&package, &request, &envelopes),
        Err(DagMlError::RuntimeValidation(_))
    ));
    let (package, request, mut envelopes) = child();
    preflight_loaded_portable_refit_replay_v3(&package, &request, &envelopes).unwrap();
    contradict(&mut envelopes);
    assert!(matches!(
        preflight_loaded_portable_refit_replay_v3(&package, &request, &envelopes),
        Err(DagMlError::RuntimeValidation(_))
    ));
}

#[test]
fn all_direct_execution_routes_refuse_before_any_callback() {
    let (source, package, request, mut envelopes) = parent();
    contradict(&mut envelopes);
    let counts = Arc::new(Counts::default());
    let registry = controllers(&package.effective_plan, &counts);
    let provider = Provider(counts.clone());
    let store = Store(counts.clone());
    let loaded = LoadedPredictor::<HandleRef>::new(package, BTreeMap::new()).unwrap();
    assert!(matches!(
        execute_attached_training_replay(AttachedTrainingReplayInput {
            source: &source,
            request: &request,
            outcome_id: "outcome:target:attached".into(),
            run_id: RunId::new("run:target:attached").unwrap(),
            controllers: &registry,
            data_provider: &provider,
            artifact_store: &store,
            data_envelopes: &envelopes,
            warnings: Vec::new(),
            diagnostics: BTreeMap::new(),
        }),
        Err(DagMlError::RuntimeValidation(_))
    ));
    counts.assert_zero();
    for phase in [Phase::Predict, Phase::Explain] {
        let mut request = request.clone();
        request.phase = phase;
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        assert!(matches!(
            execute_loaded_predictor_replay(LoadedPredictorReplayInput {
                predictor: &loaded,
                request: &request,
                outcome_id: "outcome:target:loaded".into(),
                run_id: RunId::new("run:target:loaded").unwrap(),
                controllers: &registry,
                data_provider: &provider,
                data_envelopes: &envelopes,
                warnings: Vec::new(),
                diagnostics: BTreeMap::new(),
            }),
            Err(DagMlError::RuntimeValidation(_))
        ));
        counts.assert_zero();
    }
    let (package, mut request, mut envelopes) = child();
    contradict(&mut envelopes);
    let registry = controllers(&package.outcome.effective_plan, &counts);
    for phase in [Phase::Predict, Phase::Explain] {
        request.phase = phase;
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        assert!(matches!(
            execute_loaded_portable_refit_replay_v3(LoadedPortableRefitReplayInputV3 {
                package: &package,
                request: &request,
                outcome_id: "outcome:target:child".into(),
                run_id: RunId::new("run:target:child").unwrap(),
                controllers: &registry,
                data_provider: &provider,
                data_envelopes: &envelopes,
                warnings: Vec::new(),
                diagnostics: BTreeMap::new(),
            }),
            Err(DagMlError::RuntimeValidation(_))
        ));
        counts.assert_zero();
    }
}

#[test]
fn public_owner_preserves_target_free_v2_and_absent_legacy_claims() {
    let (_, package, request, mut envelopes) = parent();
    for envelope in envelopes.values_mut() {
        let cohort = envelope.predict_cohort.as_mut().unwrap();
        cohort.role = PredictCohortRole::Inference;
        cohort.target_content_fingerprint = None;
        cohort.cohort_fingerprint = cohort.fingerprint().unwrap();
        envelope.target_content_fingerprint = None;
        envelope.validate().unwrap();
    }
    preflight_loaded_predictor_replay(&package, &request, &envelopes).unwrap();
    // Restore genuine target-bearing legacy content; it asserts no explicit
    // V2 target-name claim and remains valid through the existing owner route.
    let (_, _, _, mut legacy) = parent();
    for envelope in legacy.values_mut() {
        envelope.schema_version = 1;
        envelope.predict_cohort = None;
        envelope.validate().unwrap();
    }
    preflight_loaded_predictor_replay(&package, &request, &legacy).unwrap();
    let (package, request, mut legacy) = child();
    for envelope in legacy.values_mut() {
        envelope.schema_version = 1;
        envelope.predict_cohort = None;
        envelope.validate().unwrap();
    }
    preflight_loaded_portable_refit_replay_v3(&package, &request, &legacy).unwrap();
}

#[test]
fn canonical_port_reuses_typed_mismatch_and_requires_missing_v2_evidence_separately() {
    struct Cohort(BTreeMap<String, ExternalDataPlanEnvelope>);
    impl FacetReplayCohort for Cohort {
        fn envelopes(&self) -> &BTreeMap<String, ExternalDataPlanEnvelope> {
            &self.0
        }
    }
    let export =
        build_training_artifact_facets_from_bytes(SOURCE.as_bytes(), PACKAGE.as_bytes()).unwrap();
    let (_, _, request, envelopes) = parent();
    for absent in [false, true] {
        let mut envelopes = envelopes.clone();
        if absent {
            for envelope in envelopes.values_mut() {
                envelope.schema_version = 1;
                envelope.predict_cohort = None;
            }
        } else {
            contradict(&mut envelopes);
        }
        let result = admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(request.clone()),
            Cohort(envelopes),
        );
        if absent {
            assert!(matches!(
                result,
                Err(FacetRefusal::UnverifiedEvidence { .. })
            ));
        } else {
            assert!(matches!(result, Err(FacetRefusal::Mismatch { .. })));
        }
    }
}
