//! Generic native training and detached portable replay for synchronous hosts.
//! No feature buffers or estimator numerics cross this binding.

use super::*;
use dag_ml_core::{
    execute_loaded_predictor_replay, execute_training, ArtifactLoadMode, DataBinding,
    EnvelopeAttestedRuntimeDataProvider, ExternalDataPlanEnvelope, FittedArtifactMode,
    InMemoryArtifactStore, InMemoryDataProvider, LoadedPredictor, LoadedPredictorReplayInput,
    SampleRelationSet, TrainingDataIdentity, TrainingExecutionInput, TrainingInfluenceManifest,
    TrainingReplayRequest, TrainingRequest, EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2,
};

fn parse_contract<T: serde::de::DeserializeOwned + serde::Serialize>(
    json: &str,
    label: &str,
) -> CoreResult<T> {
    deserialize_external_contract(json, label, CoreDagMlError::CampaignValidation)
}

fn provider_for_plan(
    plan: &ExecutionPlan,
    envelopes: BTreeMap<String, ExternalDataPlanEnvelope>,
) -> CoreResult<EnvelopeAttestedRuntimeDataProvider<InMemoryDataProvider>> {
    let bindings = plan
        .node_plans
        .values()
        .flat_map(|node| node.data_bindings.iter().cloned());
    provider_for_bindings(bindings, envelopes)
}

fn provider_for_bindings(
    bindings: impl IntoIterator<Item = DataBinding>,
    envelopes: BTreeMap<String, ExternalDataPlanEnvelope>,
) -> CoreResult<EnvelopeAttestedRuntimeDataProvider<InMemoryDataProvider>> {
    let mut inner = InMemoryDataProvider::new(ControllerId::new("controller:data.provider")?);
    for envelope in envelopes.values().cloned() {
        inner.register_envelope(envelope)?;
    }
    EnvelopeAttestedRuntimeDataProvider::new(inner, bindings, envelopes)
}

fn provider_for_replay(
    plan: &ExecutionPlan,
    envelopes: BTreeMap<String, ExternalDataPlanEnvelope>,
) -> CoreResult<EnvelopeAttestedRuntimeDataProvider<InMemoryDataProvider>> {
    let mut bindings = Vec::new();
    for binding in plan
        .node_plans
        .values()
        .flat_map(|node| &node.data_bindings)
    {
        let key = dag_ml_core::data_binding_requirement_key(&binding.node_id, &binding.input_name);
        let envelope = envelopes.get(&key).ok_or_else(|| {
            CoreDagMlError::RuntimeValidation(format!(
                "training replay is missing external data envelope for `{key}`"
            ))
        })?;
        envelope.validate()?;
        // Core derives this same current-cohort relation in its cloned replay
        // plan. Attest that binding, while retaining every other saved field
        // and leaving the producer's signed package and SELECT untouched.
        let mut current = binding.clone();
        current.relation_fingerprint = envelope.relation_fingerprint.clone();
        current.validate_envelope(envelope)?;
        bindings.push(current);
    }
    provider_for_bindings(bindings, envelopes)
}

/// Sign an unsigned declaration using the same TCV1 contract as Python/C ABI.
/// Fingerprints are content seals, not cryptographic authorization signatures.
#[wasm_bindgen]
pub fn sign_training_request_json(json: &js_sys::JsString) -> Result<String, JsValue> {
    let json = crate::js_text::text_argument(json, "json")?;
    sign_training_request_json_impl(&json)
}

fn sign_training_request_json_impl(json: &str) -> Result<String, JsValue> {
    let mut request: TrainingRequest =
        parse_contract(json, "training request declaration").map_err(js_core_error)?;
    request.request_fingerprint = request.compute_fingerprint().map_err(js_core_error)?;
    request.validate().map_err(js_core_error)?;
    serde_json::to_string(&request).map_err(js_serde_error)
}

/// Derive an identity from an attested binding/envelope, without host hashing.
#[wasm_bindgen]
pub fn training_data_identity_json(
    binding_json: &js_sys::JsString,
    envelope_json: &js_sys::JsString,
) -> Result<String, JsValue> {
    let binding_json = crate::js_text::text_argument(binding_json, "binding_json")?;
    let envelope_json = crate::js_text::text_argument(envelope_json, "envelope_json")?;
    training_data_identity_json_impl(&binding_json, &envelope_json)
}

fn training_data_identity_json_impl(
    binding_json: &str,
    envelope_json: &str,
) -> Result<String, JsValue> {
    let binding = parse_contract::<DataBinding>(binding_json, "training data binding")
        .map_err(js_core_error)?;
    let envelope = parse_contract::<ExternalDataPlanEnvelope>(envelope_json, "training envelope")
        .map_err(js_core_error)?;
    let identity =
        TrainingDataIdentity::from_binding_envelope(&binding, &envelope).map_err(js_core_error)?;
    serde_json::to_string(&identity).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn sample_relation_set_fingerprint_json(json: &js_sys::JsString) -> Result<String, JsValue> {
    let json = crate::js_text::text_argument(json, "json")?;
    sample_relation_set_fingerprint_json_impl(&json)
}

fn sample_relation_set_fingerprint_json_impl(json: &str) -> Result<String, JsValue> {
    let relations: SampleRelationSet =
        parse_contract(json, "sample relations").map_err(js_core_error)?;
    relations.fingerprint().map_err(js_core_error)
}

/// Attach independently derived, target-free or external-test V2 cohort evidence.
#[wasm_bindgen]
pub fn attach_predict_cohort_to_envelope_json(
    envelope_json: &js_sys::JsString,
    cohort_request_json: &js_sys::JsString,
) -> Result<String, JsValue> {
    let envelope_json = crate::js_text::text_argument(envelope_json, "envelope_json")?;
    let cohort_request_json =
        crate::js_text::text_argument(cohort_request_json, "cohort_request_json")?;
    attach_predict_cohort_to_envelope_json_impl(&envelope_json, &cohort_request_json)
}

fn attach_predict_cohort_to_envelope_json_impl(
    envelope_json: &str,
    cohort_request_json: &str,
) -> Result<String, JsValue> {
    let mut envelope: ExternalDataPlanEnvelope =
        parse_contract(envelope_json, "data envelope").map_err(js_core_error)?;
    let request: PredictCohortConstructionRequest =
        parse_contract(cohort_request_json, "predict cohort construction request")
            .map_err(js_core_error)?;
    envelope.schema_version = EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2;
    envelope.predict_cohort = Some(request.derive().map_err(js_core_error)?);
    envelope.validate().map_err(js_core_error)?;
    serde_json::to_string(&envelope).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn sign_training_replay_request_json(json: &js_sys::JsString) -> Result<String, JsValue> {
    let json = crate::js_text::text_argument(json, "json")?;
    sign_training_replay_request_json_impl(&json)
}

fn sign_training_replay_request_json_impl(json: &str) -> Result<String, JsValue> {
    let mut request: TrainingReplayRequest =
        parse_contract(json, "training replay declaration").map_err(js_core_error)?;
    request.request_fingerprint = request.compute_fingerprint().map_err(js_core_error)?;
    request.validate().map_err(js_core_error)?;
    serde_json::to_string(&request).map_err(js_serde_error)
}

/// Execute native CV/SELECT/REFIT and capture a self-contained predictor package.
/// This synchronous lane derives influence in the core and requires every
/// fitted artifact to be native-portable. Caller-owned controllers are closed
/// by the caller, including after an exception. Exact JSON strings preserve u64.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn execute_training_json(
    request_json: &js_sys::JsString,
    data_envelopes_json: &js_sys::JsString,
    relations_json: &js_sys::JsString,
    package_id: &js_sys::JsString,
    outcome_id: &js_sys::JsString,
    run_id: &js_sys::JsString,
    bundle_id: &js_sys::JsString,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let request_json = crate::js_text::text_argument(request_json, "request_json")?;
    let data_envelopes_json =
        crate::js_text::text_argument(data_envelopes_json, "data_envelopes_json")?;
    let relations_json = crate::js_text::text_argument(relations_json, "relations_json")?;
    let package_id = crate::js_text::text_argument(package_id, "package_id")?;
    let outcome_id = crate::js_text::text_argument(outcome_id, "outcome_id")?;
    let run_id = crate::js_text::text_argument(run_id, "run_id")?;
    let bundle_id = crate::js_text::text_argument(bundle_id, "bundle_id")?;
    execute_training_json_impl(
        &request_json,
        &data_envelopes_json,
        &relations_json,
        &package_id,
        &outcome_id,
        &run_id,
        &bundle_id,
        js_invoke,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_training_json_impl(
    request_json: &str,
    data_envelopes_json: &str,
    relations_json: &str,
    package_id: &str,
    outcome_id: &str,
    run_id: &str,
    bundle_id: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let request = TrainingRequest::from_json(request_json).map_err(js_core_error)?;
    if request.options.scheduler.kind != dag_ml_core::TrainingSchedulerKind::Sequential {
        return Err(js_core_error(CoreDagMlError::RuntimeValidation(
            "synchronous WASM training requires scheduler.kind=sequential".into(),
        )));
    }
    if request.options.artifacts.fitted_artifacts != FittedArtifactMode::PortableRequired {
        return Err(js_core_error(CoreDagMlError::RuntimeValidation(
            "WASM training package requires fitted_artifacts=portable_required".into(),
        )));
    }
    let projection = request.project().map_err(js_core_error)?;
    let relations: SampleRelationSet =
        parse_contract(relations_json, "training sample relations").map_err(js_core_error)?;
    let envelopes =
        parse_contract(data_envelopes_json, "training data envelope map").map_err(js_core_error)?;
    let provider = provider_for_plan(&projection.plan, envelopes).map_err(js_core_error)?;
    let influence =
        TrainingInfluenceManifest::derive_for_projection(&projection, &request, &relations)
            .map_err(js_core_error)?;
    let controllers =
        initial_refit::controllers_for_plan(&projection.plan, js_invoke).map_err(js_core_error)?;
    let mut store = InMemoryArtifactStore::new();
    let outcome = execute_training(TrainingExecutionInput {
        request: &request,
        outcome_id: outcome_id.into(),
        run_id: RunId::new(run_id).map_err(js_core_error)?,
        bundle_id: dag_ml_core::BundleId::new(bundle_id).map_err(js_core_error)?,
        controllers: &controllers,
        data_provider: &provider,
        relations: &relations,
        training_influence: &influence,
        artifact_store: &mut store,
        warnings: Vec::new(),
        diagnostics: BTreeMap::new(),
    })
    .map_err(js_core_error)?;
    let package = outcome
        .to_portable_predictor_package(
            package_id,
            FittedArtifactMode::PortableRequired,
            ArtifactLoadMode::NativePortable,
        )
        .map_err(js_core_error)?;
    let package_json = serde_json::to_string(&package).map_err(js_serde_error)?;
    let outcome_json = serde_json::to_string(&outcome).map_err(js_serde_error)?;
    serde_json::to_string(&serde_json::json!({
        "portable_predictor_package_json": package_json,
        "training_outcome_json": outcome_json,
    }))
    .map_err(js_serde_error)
}

/// Replay a detached native-portable package; the core hydrates and releases
/// RAW artifacts for this invocation, including on failed prediction.
/// The trusted runtime manifests must match the package before any callback.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn replay_training_package_json(
    package_json: &js_sys::JsString,
    request_json: &js_sys::JsString,
    data_envelopes_json: &js_sys::JsString,
    trusted_controller_manifests_json: &js_sys::JsString,
    outcome_id: &js_sys::JsString,
    run_id: &js_sys::JsString,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let package_json = crate::js_text::text_argument(package_json, "package_json")?;
    let request_json = crate::js_text::text_argument(request_json, "request_json")?;
    let data_envelopes_json =
        crate::js_text::text_argument(data_envelopes_json, "data_envelopes_json")?;
    let trusted_controller_manifests_json = crate::js_text::text_argument(
        trusted_controller_manifests_json,
        "trusted_controller_manifests_json",
    )?;
    let outcome_id = crate::js_text::text_argument(outcome_id, "outcome_id")?;
    let run_id = crate::js_text::text_argument(run_id, "run_id")?;
    replay_training_package_json_impl(
        &package_json,
        &request_json,
        &data_envelopes_json,
        &trusted_controller_manifests_json,
        &outcome_id,
        &run_id,
        js_invoke,
    )
}

#[allow(clippy::too_many_arguments)]
fn replay_training_package_json_impl(
    package_json: &str,
    request_json: &str,
    data_envelopes_json: &str,
    trusted_controller_manifests_json: &str,
    outcome_id: &str,
    run_id: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let package = PortablePredictorPackage::from_json(package_json).map_err(js_core_error)?;
    if package.fitted_artifact_mode != FittedArtifactMode::PortableRequired
        || package
            .artifact_bindings
            .iter()
            .any(|binding| binding.load_mode != ArtifactLoadMode::NativePortable)
    {
        return Err(js_core_error(CoreDagMlError::RuntimeValidation(
            "WASM detached replay requires only native-portable artifacts".into(),
        )));
    }
    let trusted = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&package.effective_plan, &trusted)
        .map_err(js_core_error)?;
    let request = TrainingReplayRequest::from_json(request_json).map_err(js_core_error)?;
    let envelopes: BTreeMap<String, ExternalDataPlanEnvelope> =
        parse_contract(data_envelopes_json, "replay data envelope map").map_err(js_core_error)?;
    let provider =
        provider_for_replay(&package.effective_plan, envelopes.clone()).map_err(js_core_error)?;
    let controllers = initial_refit::controllers_for_plan(&package.effective_plan, js_invoke)
        .map_err(js_core_error)?;
    let predictor =
        LoadedPredictor::<HandleRef>::new(package, BTreeMap::new()).map_err(js_core_error)?;
    let replay = execute_loaded_predictor_replay(LoadedPredictorReplayInput {
        predictor: &predictor,
        request: &request,
        outcome_id: outcome_id.into(),
        run_id: RunId::new(run_id).map_err(js_core_error)?,
        controllers: &controllers,
        data_provider: &provider,
        data_envelopes: &envelopes,
        warnings: Vec::new(),
        diagnostics: BTreeMap::new(),
    })
    .map_err(js_core_error)?;
    serde_json::to_string(&replay).map_err(js_serde_error)
}

#[cfg(test)]
mod replay_provider_tests {
    use super::*;
    use dag_ml_core::{Phase, PredictCohort, PredictCohortRole, RuntimeDataProvider};

    fn current_target_free_fixture() -> (
        PortablePredictorPackage,
        BTreeMap<String, ExternalDataPlanEnvelope>,
    ) {
        let package = PortablePredictorPackage::from_json(include_str!(
            "../../../examples/fixtures/training/portable_predictor_package.v1.json"
        ))
        .unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../examples/fixtures/training/replay/training_replay_input_envelopes.v1.json"
        ))
        .unwrap();
        let mut envelopes: BTreeMap<String, ExternalDataPlanEnvelope> =
            serde_json::from_value(fixture["envelopes"].clone()).unwrap();
        // These requirements share one raw IO plan and current physical
        // cohort. Register its single content/metadata identity consistently.
        let current = envelopes.values().next().unwrap().clone();
        for envelope in envelopes.values_mut() {
            *envelope = current.clone();
            envelope.schema_version = EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION_V2;
            envelope.target_content_fingerprint = None;
            envelope.predict_cohort = Some(
                PredictCohort::from_relations(
                    PredictCohortRole::Inference,
                    envelope.coordinator_relations.clone().unwrap(),
                    package.output_bindings[0].target_names.clone(),
                    envelope.data_content_fingerprint.clone().unwrap(),
                    None,
                )
                .unwrap(),
            );
            envelope.validate().unwrap();
        }
        (package, envelopes)
    }

    #[test]
    fn replay_provider_attests_current_target_free_relations_without_changing_package() {
        let (package, envelopes) = current_target_free_fixture();
        let original = serde_json::to_string(&package).unwrap();
        let saved_provider = provider_for_plan(&package.effective_plan, envelopes.clone()).unwrap();
        let provider = provider_for_replay(&package.effective_plan, envelopes.clone()).unwrap();
        for saved in package
            .effective_plan
            .node_plans
            .values()
            .flat_map(|node| &node.data_bindings)
        {
            let key = dag_ml_core::data_binding_requirement_key(&saved.node_id, &saved.input_name);
            let mut current = saved.clone();
            current.relation_fingerprint = envelopes[&key].relation_fingerprint.clone();
            assert_ne!(current.relation_fingerprint, saved.relation_fingerprint);
            let previous_error = saved_provider
                .coordinator_relations(&current)
                .unwrap_err()
                .to_string();
            assert!(previous_error.contains("does not exactly match"));
            assert_eq!(
                provider.coordinator_relations(&current).unwrap(),
                envelopes[&key].coordinator_relations
            );
            assert_eq!(
                provider.predict_cohort(&current, Phase::Predict).unwrap(),
                envelopes[&key].predict_cohort
            );
            assert!(provider.training_data_identity(&current).unwrap().is_none());
            assert!(provider.coordinator_relations(saved).is_err());
            let mut foreign = current;
            foreign.schema_fingerprint = "a".repeat(64);
            assert!(provider.coordinator_relations(&foreign).is_err());
        }
        assert_eq!(serde_json::to_string(&package).unwrap(), original);
        assert!(provider.inner().handle_records().is_empty());
    }

    #[test]
    fn replay_provider_refuses_missing_envelope_before_delegation() {
        let (package, mut envelopes) = current_target_free_fixture();
        let key = envelopes.keys().next().unwrap().clone();
        envelopes.remove(&key);
        let error = provider_for_replay(&package.effective_plan, envelopes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing external data envelope"));
        assert!(error.contains(&key));
    }

    #[test]
    fn replay_provider_refuses_wrong_envelope_contract_before_delegation() {
        let (package, envelopes) = current_target_free_fixture();
        let key = envelopes.keys().next().unwrap().clone();
        for field in ["schema", "plan"] {
            let mut wrong = envelopes.clone();
            let envelope = wrong.get_mut(&key).unwrap();
            if field == "schema" {
                envelope.schema_fingerprint = "a".repeat(64);
            } else {
                envelope.plan_fingerprint = "a".repeat(64);
            }
            assert!(provider_for_replay(&package.effective_plan, wrong).is_err());
        }
    }

    #[test]
    fn rust_wasm_replay_delegation_refuses_explicit_target_claim_before_callbacks() {
        use dag_ml_core::{
            DataMaterializationRequest, DataViewRequest, HandleKind, NodeResult, NodeTask,
            RuntimeController, RuntimeControllerRegistry,
        };
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        struct Controller {
            id: ControllerId,
            calls: Arc<AtomicUsize>,
        }
        impl RuntimeController for Controller {
            fn controller_id(&self) -> &ControllerId {
                &self.id
            }
            fn invoke(&self, _: &NodeTask) -> CoreResult<NodeResult> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Err(CoreDagMlError::RuntimeValidation(
                    "unexpected test controller callback".into(),
                ))
            }
        }
        struct Provider<'a> {
            inner: &'a EnvelopeAttestedRuntimeDataProvider<InMemoryDataProvider>,
            calls: Arc<AtomicUsize>,
        }
        impl RuntimeDataProvider for Provider<'_> {
            fn materialize(&self, request: &DataMaterializationRequest) -> CoreResult<HandleRef> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.materialize(request)
            }
            fn make_view(&self, request: &DataViewRequest) -> CoreResult<HandleRef> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.make_view(request)
            }
            fn coordinator_relations(
                &self,
                binding: &DataBinding,
            ) -> CoreResult<Option<dag_ml_core::SampleRelationSet>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.coordinator_relations(binding)
            }
            fn predict_cohort(
                &self,
                binding: &DataBinding,
                phase: Phase,
            ) -> CoreResult<Option<PredictCohort>> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                self.inner.predict_cohort(binding, phase)
            }
        }
        let (package, mut envelopes) = current_target_free_fixture();
        let mut request = TrainingReplayRequest {
            schema_version: 1,
            request_id: "replay:wasm:target-claim".into(),
            source_outcome_fingerprint: package.training_outcome.outcome_fingerprint.clone(),
            phase: Phase::Predict,
            data_envelope_keys: envelopes.keys().cloned().collect(),
            output_binding_ids: package
                .output_bindings
                .iter()
                .map(|binding| binding.binding_id.clone())
                .collect(),
            request_fingerprint: String::new(),
        };
        request.output_binding_ids.sort();
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        dag_ml_core::preflight_loaded_predictor_replay(&package, &request, &envelopes).unwrap();
        let valid_envelopes = envelopes.clone();
        for envelope in envelopes.values_mut() {
            let cohort = envelope.predict_cohort.as_mut().unwrap();
            cohort.target_names = vec!["contradictory-target".into()];
            cohort.cohort_fingerprint = cohort.fingerprint().unwrap();
            envelope.validate().unwrap();
        }
        let inner = provider_for_replay(&package.effective_plan, envelopes.clone()).unwrap();
        let data_calls = Arc::new(AtomicUsize::new(0));
        let model_calls = Arc::new(AtomicUsize::new(0));
        let provider = Provider {
            inner: &inner,
            calls: data_calls.clone(),
        };
        let mut controllers = RuntimeControllerRegistry::new();
        for id in package
            .effective_plan
            .node_plans
            .values()
            .map(|node| node.controller_id.clone())
            .collect::<std::collections::BTreeSet<_>>()
        {
            controllers
                .register(Box::new(Controller {
                    id,
                    calls: model_calls.clone(),
                }))
                .unwrap();
        }
        // Explicit test-only trust boundary for this existing mock sidecar
        // fixture. No serializer bytes or numerical model are hydrated here.
        let loaded = package
            .load_with(|record| {
                Ok(HandleRef {
                    handle: 1,
                    kind: HandleKind::Artifact,
                    owner_controller: record.controller_id.clone(),
                })
            })
            .unwrap();
        let valid_inner =
            provider_for_replay(&loaded.package().effective_plan, valid_envelopes.clone()).unwrap();
        let control_provider = Provider {
            inner: &valid_inner,
            calls: data_calls.clone(),
        };
        // Valid inputs must reach an actual callback through this same Rust
        // binding/provider path; the trap is not numerical qualification.
        assert!(execute_loaded_predictor_replay(LoadedPredictorReplayInput {
            predictor: &loaded,
            request: &request,
            outcome_id: "outcome:wasm:control".into(),
            run_id: RunId::new("run:wasm:control").unwrap(),
            controllers: &controllers,
            data_provider: &control_provider,
            data_envelopes: &valid_envelopes,
            warnings: Vec::new(),
            diagnostics: BTreeMap::new(),
        })
        .is_err());
        assert!(model_calls.load(Ordering::SeqCst) + data_calls.load(Ordering::SeqCst) > 0);
        model_calls.store(0, Ordering::SeqCst);
        data_calls.store(0, Ordering::SeqCst);
        let result = execute_loaded_predictor_replay(LoadedPredictorReplayInput {
            predictor: &loaded,
            request: &request,
            outcome_id: "outcome:wasm:targets".into(),
            run_id: RunId::new("run:wasm:targets").unwrap(),
            controllers: &controllers,
            data_provider: &provider,
            data_envelopes: &envelopes,
            warnings: Vec::new(),
            diagnostics: BTreeMap::new(),
        });
        assert!(matches!(result, Err(CoreDagMlError::RuntimeValidation(_))));
        assert_eq!(model_calls.load(Ordering::SeqCst), 0);
        assert_eq!(data_calls.load(Ordering::SeqCst), 0);
        assert!(inner.inner().handle_records().is_empty());
    }
}
