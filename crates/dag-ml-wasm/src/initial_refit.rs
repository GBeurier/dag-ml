//! Browser/Node bindings for a no-splitter initial REFIT and later PREDICT replay.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

use dag_ml_core::{
    execute_initial_full_refit, execute_initial_full_refit_prediction, ArtifactId,
    ArtifactLoadMode, ExplicitPhaseDataProvider, ExternalDataPlanEnvelope, HandleRef,
    InMemoryArtifactStore, InitialFullRefitExecutionInput, InitialRefitReplayInput,
    InitialRefitScheduler,
};

fn controllers_for_plan(
    plan: &ExecutionPlan,
    js_invoke: &js_sys::Function,
) -> CoreResult<RuntimeControllerRegistry> {
    let mut controllers = RuntimeControllerRegistry::new();
    for manifest in plan.controller_manifests.values() {
        controllers.register(Box::new(JsRuntimeController {
            id: manifest.controller_id.clone(),
            js_invoke: js_invoke.clone(),
        }))?;
    }
    Ok(controllers)
}

/// Execute a no-splitter REFIT once and return the closed package and node evidence.
/// The synchronous JS callback owns host-sidecar operator state by artifact ID.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn execute_initial_full_refit_json(
    plan_json: &str,
    trusted_controller_manifests_json: &str,
    envelope_json: &str,
    training_sample_ids_json: &str,
    package_id: &str,
    run_id: &str,
    root_seed: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let plan = ExecutionPlan::from_json(plan_json).map_err(js_core_error)?;
    let trusted = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&plan, &trusted).map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope = deserialize_external_contract(
        envelope_json,
        "initial full-refit data envelope",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(js_core_error)?;
    let ids: Vec<SampleId> = deserialize_external_contract(
        training_sample_ids_json,
        "initial full-refit training IDs",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(js_core_error)?;
    let provider = ExplicitPhaseDataProvider::new(
        ControllerId::new("controller:data.provider").map_err(js_core_error)?,
        envelope.clone(),
        Some(ids.clone()),
    )
    .map_err(js_core_error)?;
    let controllers = controllers_for_plan(&plan, js_invoke).map_err(js_core_error)?;
    let root_seed = root_seed.parse::<u64>().map_err(|_| {
        js_core_error(CoreDagMlError::CampaignValidation(
            "initial full-refit root seed must be a decimal u64 string".into(),
        ))
    })?;
    let execution = execute_initial_full_refit(InitialFullRefitExecutionInput {
        package_id: package_id.to_owned(),
        run_id: RunId::new(run_id).map_err(js_core_error)?,
        plan: &plan,
        training_envelope: &envelope,
        training_sample_ids: &ids,
        controllers: &controllers,
        data_provider: &provider,
        root_seed: Some(root_seed),
        resource_limits: None,
        scheduler: InitialRefitScheduler::Sequential,
    })
    .map_err(js_core_error)?;
    // JSON.parse rounds u64 values above 2^53. Expose exact package bytes
    // alongside the decoded object so JavaScript can replay a sealed package.
    let package_json = serde_json::to_string(&execution.package).map_err(js_serde_error)?;
    serde_json::to_string(&serde_json::json!({
        "initial_full_refit_package": execution.package,
        "initial_full_refit_package_json": package_json,
        "node_results": execution.results, "scores": execution.scores,
    }))
    .map_err(js_serde_error)
}

/// Replay selected package outputs on a fresh V2 cohort without training.
/// Raw native payloads are hydrated from the package; host sidecars use exact
/// caller-owned artifact handles registered for this invocation only.
#[wasm_bindgen]
pub fn replay_initial_full_refit_json(
    package_json: &str,
    envelope_json: &str,
    output_ids_json: &str,
    artifact_handles_json: &str,
    run_id: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let package = InitialFullRefitPackage::from_json(package_json).map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope = deserialize_external_contract(
        envelope_json,
        "initial full-refit predict envelope",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(js_core_error)?;
    let output_ids: Vec<String> = deserialize_external_contract(
        output_ids_json,
        "initial full-refit output IDs",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(js_core_error)?;
    let handles: BTreeMap<ArtifactId, HandleRef> = deserialize_external_contract(
        artifact_handles_json,
        "initial full-refit sidecar handles",
        CoreDagMlError::CampaignValidation,
    )
    .map_err(js_core_error)?;
    let expected = package
        .artifacts
        .iter()
        .filter(|artifact| artifact.load_mode == ArtifactLoadMode::HostSidecar)
        .map(|artifact| &artifact.record.artifact.id)
        .collect::<BTreeSet<_>>();
    if handles.keys().collect::<BTreeSet<_>>() != expected {
        return Err(js_core_error(CoreDagMlError::RuntimeValidation(
            "initial full-refit handles must exactly cover host-sidecar artifacts".into(),
        )));
    }
    let provider = ExplicitPhaseDataProvider::new(
        ControllerId::new("controller:data.provider").map_err(js_core_error)?,
        envelope.clone(),
        None,
    )
    .map_err(js_core_error)?;
    let controllers =
        controllers_for_plan(&package.effective_plan, js_invoke).map_err(js_core_error)?;
    let mut store = InMemoryArtifactStore::new();
    for artifact in &package.artifacts {
        if artifact.load_mode == ArtifactLoadMode::HostSidecar {
            store
                .register(
                    &artifact.record,
                    handles[&artifact.record.artifact.id].clone(),
                )
                .map_err(js_core_error)?;
        }
    }
    let replay = execute_initial_full_refit_prediction(InitialRefitReplayInput {
        package: &package,
        envelope: &envelope,
        output_ids: &output_ids,
        run_id: RunId::new(run_id).map_err(js_core_error)?,
        controllers: &controllers,
        data_provider: &provider,
        artifact_store: &store,
    })
    .map_err(js_core_error)?;
    serde_json::to_string(&serde_json::json!({
        "replay_outcome": replay.outcome, "node_results": replay.results,
    }))
    .map_err(js_serde_error)
}
