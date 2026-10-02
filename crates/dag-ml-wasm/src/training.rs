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

/// Assemble the canonical opaque payload set consumed by Core's bounded ZIP writer.
#[wasm_bindgen]
pub fn build_archive_v2_native_portable_payloads_json(
    archive_id: &str,
    outcome_json: &str,
    package_json: &str,
) -> Result<String, JsValue> {
    let outcome = dag_ml_core::TrainingOutcome::from_json(outcome_json).map_err(js_core_error)?;
    let package = PortablePredictorPackage::from_json(package_json).map_err(js_core_error)?;
    let payloads =
        dag_ml_core::build_archive_v2_native_portable_payloads(archive_id, &outcome, &package)
            .map_err(js_core_error)?;
    serde_json::to_string(&serde_json::json!({
        "manifest": payloads.manifest, "members": payloads.members
    }))
    .map_err(js_serde_error)
}

/// Validate opaque archive members before replay can invoke a controller.
#[wasm_bindgen]
pub fn validate_archive_v2_portable_payloads_json(
    manifest_json: &str,
    package_json: &str,
    members_json: &str,
) -> Result<String, JsValue> {
    let manifest = parse_contract::<serde_json::Value>(manifest_json, "Archive V2 manifest")
        .map_err(js_core_error)?;
    let package = PortablePredictorPackage::from_json(package_json).map_err(js_core_error)?;
    let members = parse_contract::<BTreeMap<String, Vec<u8>>>(members_json, "Archive V2 members")
        .map_err(js_core_error)?;
    dag_ml_core::validate_archive_v2_portable_payloads(&manifest, &package, &members)
        .map_err(js_core_error)?;
    Ok("{\"valid\":true}".to_string())
}

/// Sign an unsigned declaration using the same TCV1 contract as Python/C ABI.
/// Fingerprints are content seals, not cryptographic authorization signatures.
#[wasm_bindgen]
pub fn sign_training_request_json(json: &str) -> Result<String, JsValue> {
    let mut request: TrainingRequest =
        parse_contract(json, "training request declaration").map_err(js_core_error)?;
    request.request_fingerprint = request.compute_fingerprint().map_err(js_core_error)?;
    request.validate().map_err(js_core_error)?;
    serde_json::to_string(&request).map_err(js_serde_error)
}

/// Derive an identity from an attested binding/envelope, without host hashing.
#[wasm_bindgen]
pub fn training_data_identity_json(
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
pub fn sample_relation_set_fingerprint_json(json: &str) -> Result<String, JsValue> {
    let relations: SampleRelationSet =
        parse_contract(json, "sample relations").map_err(js_core_error)?;
    relations.fingerprint().map_err(js_core_error)
}

/// Attach independently derived, target-free or external-test V2 cohort evidence.
#[wasm_bindgen]
pub fn attach_predict_cohort_to_envelope_json(
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
pub fn sign_training_replay_request_json(json: &str) -> Result<String, JsValue> {
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
                    vec!["y".into()],
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
}
