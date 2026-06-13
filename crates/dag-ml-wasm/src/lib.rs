//! Browser-friendly bindings for DAG-ML JSON contracts.
//!
//! The WASM surface exposes validation, DSL compilation and plan construction
//! over UTF-8 JSON strings, plus an `execute_campaign_phase_json` entry point
//! that drives the in-process [`SequentialScheduler`] with host operators
//! supplied as a synchronous JavaScript callback. Feature matrices, fitted
//! models and data buffers still never cross this boundary — the JS controller
//! resolves those out-of-band, the core only routes ids, predictions and
//! lineage (the same ownership boundary as the C ABI host bridge).

use serde::de::DeserializeOwned;
use wasm_bindgen::prelude::*;

use std::collections::BTreeMap;

use dag_ml_core::{
    build_execution_plan, compile_pipeline_dsl, compile_pipeline_dsl_with_generation,
    compile_pipeline_dsl_with_generation_and_controller_registry, fold_set_fingerprint,
    parse_pipeline_dsl_json, select_candidate, select_candidate_groups, CampaignSpec,
    CandidateScore, ControllerManifest, ControllerRegistry, DagMlError as CoreDagMlError,
    ExecutionBundle, ExecutionPlan, FoldSet, GraphSpec, KFoldSpec, SampleId, SelectionPolicy,
    StratifiedKFoldSpec,
};
use dag_ml_core::{
    ControllerId, NodeResult, NodeTask, Phase, Result as CoreResult, RunContext, RunId,
    RuntimeController, RuntimeControllerRegistry, SequentialScheduler,
};

const SHARED_FOLD_SET_FINGERPRINT: &str =
    "54d3185d6c628ef0df848828a8d8ae650222a283a78bbd3ab3bc2256f222c05c";

#[wasm_bindgen]
pub fn dag_ml_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[wasm_bindgen]
pub fn contract_manifest_json() -> Result<String, JsValue> {
    serde_json::to_string(&contract_manifest()).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn validate_graph_json(json: &str) -> Result<(), JsValue> {
    validate_json::<GraphSpec>(json, GraphSpec::validate)
}

#[wasm_bindgen]
pub fn validate_campaign_json(json: &str) -> Result<(), JsValue> {
    validate_json::<CampaignSpec>(json, CampaignSpec::validate)
}

#[wasm_bindgen]
pub fn validate_controller_manifest_json(json: &str) -> Result<(), JsValue> {
    validate_json::<ControllerManifest>(json, ControllerManifest::validate)
}

#[wasm_bindgen]
pub fn validate_controller_manifest_list_json(json: &str) -> Result<(), JsValue> {
    controller_registry_from_json(json).map(|_| ())
}

#[wasm_bindgen]
pub fn validate_pipeline_dsl_json(json: &str) -> Result<(), JsValue> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(js_core_error)?;
    compile_pipeline_dsl_with_generation(&spec)
        .map(|_| ())
        .map_err(js_core_error)
}

#[wasm_bindgen]
pub fn validate_execution_plan_json(json: &str) -> Result<(), JsValue> {
    validate_json::<ExecutionPlan>(json, ExecutionPlan::validate)
}

#[wasm_bindgen]
pub fn validate_execution_bundle_json(json: &str) -> Result<(), JsValue> {
    validate_json::<ExecutionBundle>(json, ExecutionBundle::validate)
}

#[wasm_bindgen]
pub fn validate_fold_set_json(json: &str) -> Result<(), JsValue> {
    validate_json::<FoldSet>(json, FoldSet::validate)
}

#[wasm_bindgen]
pub fn fold_set_fingerprint_json(json: &str) -> Result<String, JsValue> {
    let fold_set = parse_and_validate::<FoldSet>(json, FoldSet::validate)?;
    fold_set_fingerprint(&fold_set).map_err(js_core_error)
}

/// Build a K-fold `FoldSet` from a `KFoldSpec` JSON + a JSON array of sample ids.
/// dag-ml owns the split — the host stops building folds itself.
#[wasm_bindgen]
pub fn kfold_split_json(
    spec_json: &str,
    sample_ids_json: &str,
    id: &str,
) -> Result<String, JsValue> {
    let spec: KFoldSpec = serde_json::from_str(spec_json).map_err(js_serde_error)?;
    let samples: Vec<SampleId> = serde_json::from_str(sample_ids_json).map_err(js_serde_error)?;
    let fold_set = spec.split(id, &samples).map_err(js_core_error)?;
    serde_json::to_string(&fold_set).map_err(js_serde_error)
}

/// Build a stratified K-fold `FoldSet`: same OOF-once guarantee as K-fold, but
/// balanced by a per-sample class label. `strata_json` is a JSON object mapping
/// sample id → class label (identity-keyed metadata, never feature values).
#[wasm_bindgen]
pub fn stratified_kfold_split_json(
    spec_json: &str,
    sample_ids_json: &str,
    strata_json: &str,
    id: &str,
) -> Result<String, JsValue> {
    let spec: StratifiedKFoldSpec = serde_json::from_str(spec_json).map_err(js_serde_error)?;
    let samples: Vec<SampleId> = serde_json::from_str(sample_ids_json).map_err(js_serde_error)?;
    let strata: BTreeMap<SampleId, String> =
        serde_json::from_str(strata_json).map_err(js_serde_error)?;
    let fold_set = spec.split(id, &samples, &strata).map_err(js_core_error)?;
    serde_json::to_string(&fold_set).map_err(js_serde_error)
}

/// Rank candidate variants and return the winner — the SELECT phase for in-browser
/// generators/finetune. Selection stays in dag-ml (deterministic argmin/argmax +
/// id tie-break), not the host. `policy_json` = SelectionPolicy, `candidates_json`
/// = `CandidateScore` array. With `groups_json` (group id to candidate ids) returns a
/// {group → SelectionDecision} map; otherwise a single SelectionDecision.
#[wasm_bindgen]
pub fn select_candidates_json(
    policy_json: &str,
    candidates_json: &str,
    groups_json: Option<String>,
) -> Result<String, JsValue> {
    let policy: SelectionPolicy = serde_json::from_str(policy_json).map_err(js_serde_error)?;
    let candidates: Vec<CandidateScore> =
        serde_json::from_str(candidates_json).map_err(js_serde_error)?;
    match groups_json {
        Some(groups_json) => {
            let groups: BTreeMap<String, Vec<String>> =
                serde_json::from_str(&groups_json).map_err(js_serde_error)?;
            let decisions =
                select_candidate_groups(&policy, &candidates, &groups).map_err(js_core_error)?;
            serde_json::to_string(&decisions).map_err(js_serde_error)
        }
        None => {
            let decision = select_candidate(&policy, &candidates).map_err(js_core_error)?;
            serde_json::to_string(&decision).map_err(js_serde_error)
        }
    }
}

#[wasm_bindgen]
pub fn compile_pipeline_dsl_graph_json(json: &str) -> Result<String, JsValue> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(js_core_error)?;
    let graph = compile_pipeline_dsl(&spec).map_err(js_core_error)?;
    serde_json::to_string(&graph).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn compile_pipeline_dsl_artifact_json(json: &str) -> Result<String, JsValue> {
    let spec = parse_pipeline_dsl_json(json.as_bytes()).map_err(js_core_error)?;
    let artifact = compile_pipeline_dsl_with_generation(&spec).map_err(js_core_error)?;
    serde_json::to_string(&artifact).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn compile_pipeline_dsl_artifact_with_controllers_json(
    dsl_json: &str,
    controller_manifests_json: &str,
) -> Result<String, JsValue> {
    let spec = parse_pipeline_dsl_json(dsl_json.as_bytes()).map_err(js_core_error)?;
    let registry = controller_registry_from_json(controller_manifests_json)?;
    let artifact = compile_pipeline_dsl_with_generation_and_controller_registry(&spec, &registry)
        .map_err(js_core_error)?;
    serde_json::to_string(&artifact).map_err(js_serde_error)
}

#[wasm_bindgen]
pub fn build_execution_plan_json(
    plan_id: &str,
    graph_json: &str,
    campaign_json: &str,
    controller_manifests_json: &str,
) -> Result<String, JsValue> {
    let graph = parse_and_validate::<GraphSpec>(graph_json, GraphSpec::validate)?;
    let campaign = parse_and_validate::<CampaignSpec>(campaign_json, CampaignSpec::validate)?;
    let registry = controller_registry_from_json(controller_manifests_json)?;
    let plan = build_execution_plan(plan_id.to_string(), graph, campaign, &registry)
        .map_err(js_core_error)?;
    serde_json::to_string(&plan).map_err(js_serde_error)
}

fn validate_json<T>(
    json: &str,
    validate: impl FnOnce(&T) -> dag_ml_core::Result<()>,
) -> Result<(), JsValue>
where
    T: DeserializeOwned,
{
    parse_and_validate::<T>(json, validate).map(|_| ())
}

fn parse_and_validate<T>(
    json: &str,
    validate: impl FnOnce(&T) -> dag_ml_core::Result<()>,
) -> Result<T, JsValue>
where
    T: DeserializeOwned,
{
    let value = serde_json::from_str::<T>(json).map_err(js_serde_error)?;
    validate(&value).map_err(js_core_error)?;
    Ok(value)
}

fn controller_registry_from_json(json: &str) -> Result<ControllerRegistry, JsValue> {
    let manifests =
        serde_json::from_str::<Vec<ControllerManifest>>(json).map_err(js_serde_error)?;
    let mut registry = ControllerRegistry::new();
    for manifest in manifests {
        registry.register(manifest).map_err(js_core_error)?;
    }
    Ok(registry)
}

fn contract_manifest() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "crate": "dag-ml",
        "package": "dag-ml",
        "version": env!("CARGO_PKG_VERSION"),
        "surface": "json-contract-bindings",
        "contracts": [
            {"id": "graph_spec", "version": 1},
            {"id": "campaign_spec", "version": 1},
            {"id": "controller_manifest", "version": 1},
            {"id": "pipeline_dsl", "version": 1},
            {"id": "execution_plan", "version": 1},
            {"id": "execution_bundle", "version": 1},
            {"id": "fold_set", "version": 1}
        ],
        "capabilities": [
            "validate_json_contracts",
            "compile_pipeline_dsl",
            "compile_pipeline_dsl_with_generation",
            "compile_pipeline_dsl_with_controller_registry",
            "build_execution_plan",
            "fold_set_fingerprint",
            "structured_error_descriptors"
        ],
        "shared": {
            "fold_set_fixture_fingerprint": SHARED_FOLD_SET_FINGERPRINT
        },
        "python_exports": [
            "version",
            "contract_manifest_json",
            "validate_graph_json",
            "validate_campaign_json",
            "validate_controller_manifest_json",
            "validate_controller_manifest_list_json",
            "validate_pipeline_dsl_json",
            "validate_execution_plan_json",
            "validate_execution_bundle_json",
            "validate_fold_set_json",
            "fold_set_fingerprint_json",
            "compile_pipeline_dsl_graph_json",
            "compile_pipeline_dsl_artifact_json",
            "compile_pipeline_dsl_artifact_with_controllers_json",
            "build_execution_plan_json"
        ],
        "wasm_exports": [
            "dag_ml_version",
            "contract_manifest_json",
            "validate_graph_json",
            "validate_campaign_json",
            "validate_controller_manifest_json",
            "validate_controller_manifest_list_json",
            "validate_pipeline_dsl_json",
            "validate_execution_plan_json",
            "validate_execution_bundle_json",
            "validate_fold_set_json",
            "fold_set_fingerprint_json",
            "compile_pipeline_dsl_graph_json",
            "compile_pipeline_dsl_artifact_json",
            "compile_pipeline_dsl_artifact_with_controllers_json",
            "build_execution_plan_json",
            "kfold_split_json",
            "stratified_kfold_split_json",
            "select_candidates_json",
            "execute_campaign_phase_json"
        ]
    })
}

fn js_serde_error(error: serde_json::Error) -> JsValue {
    js_core_error(CoreDagMlError::Serialization(error))
}

fn js_core_error(error: CoreDagMlError) -> JsValue {
    let payload = error
        .descriptor_json()
        .unwrap_or_else(|_| error.to_string());
    JsValue::from_str(&payload)
}

// ---------------------------------------------------------------------------
// In-browser execution: drive the SequentialScheduler with a JS controller.
// ---------------------------------------------------------------------------

/// A [`RuntimeController`] backed by a synchronous JavaScript callback.
///
/// `invoke` serializes the `NodeTask` to JSON, calls
/// `js_invoke(controller_id, task_json)` and parses the returned `NodeResult`
/// JSON. The host (JS) resolves feature matrices and fitted models out-of-band
/// — the core never sees them.
struct JsRuntimeController {
    id: ControllerId,
    js_invoke: js_sys::Function,
}

// SAFETY: wasm32 is single-threaded and `SequentialScheduler` never moves the
// controller across threads; the held `js_sys::Function` is only ever called on
// the same (only) thread. Mirrors the `CAbiRuntimeController` precedent in
// `dag-ml-capi`.
unsafe impl Send for JsRuntimeController {}
unsafe impl Sync for JsRuntimeController {}

impl RuntimeController for JsRuntimeController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> CoreResult<NodeResult> {
        let task_json = serde_json::to_string(task).map_err(CoreDagMlError::Serialization)?;
        let returned = self
            .js_invoke
            .call2(
                &JsValue::NULL,
                &JsValue::from_str(self.id.as_str()),
                &JsValue::from_str(&task_json),
            )
            .map_err(|err| {
                CoreDagMlError::RuntimeValidation(format!(
                    "JS controller `{}` threw: {err:?}",
                    self.id
                ))
            })?;
        let result_json = returned.as_string().ok_or_else(|| {
            CoreDagMlError::RuntimeValidation(format!(
                "JS controller `{}` must return a NodeResult JSON string",
                self.id
            ))
        })?;
        serde_json::from_str::<NodeResult>(&result_json).map_err(CoreDagMlError::Serialization)
    }
}

fn parse_phase(phase: &str) -> Result<Phase, JsValue> {
    match phase {
        "FIT_CV" => Ok(Phase::FitCv),
        "SELECT" => Ok(Phase::Select),
        "REFIT" => Ok(Phase::Refit),
        "PREDICT" => Ok(Phase::Predict),
        "EXPLAIN" => Ok(Phase::Explain),
        other => Err(JsValue::from_str(&format!("unknown phase `{other}`"))),
    }
}

/// Execute one phase of a campaign with the in-process [`SequentialScheduler`],
/// invoking host operators through the supplied JS callback.
///
/// - `graph_json` / `campaign_json` / `controller_manifests_json`: the same
///   inputs as [`build_execution_plan_json`]. The campaign's
///   `split_invocation.fold_set` drives the FIT_CV fold loop.
/// - `js_invoke`: `(controllerId: string, taskJson: string) => nodeResultJson: string`,
///   **synchronous** (no `await` across this boundary).
///
/// Returns the phase's `Vec<NodeResult>` as JSON (predictions + lineage).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn execute_campaign_phase_json(
    plan_id: &str,
    graph_json: &str,
    campaign_json: &str,
    controller_manifests_json: &str,
    run_id: &str,
    root_seed: u32,
    phase: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let graph = parse_and_validate::<GraphSpec>(graph_json, GraphSpec::validate)?;
    let campaign = parse_and_validate::<CampaignSpec>(campaign_json, CampaignSpec::validate)?;
    let manifests = serde_json::from_str::<Vec<ControllerManifest>>(controller_manifests_json)
        .map_err(js_serde_error)?;

    let mut registry = ControllerRegistry::new();
    for manifest in &manifests {
        registry.register(manifest.clone()).map_err(js_core_error)?;
    }
    let plan = build_execution_plan(plan_id.to_string(), graph, campaign, &registry)
        .map_err(js_core_error)?;

    let mut controllers = RuntimeControllerRegistry::new();
    for manifest in &manifests {
        controllers
            .register(Box::new(JsRuntimeController {
                id: manifest.controller_id.clone(),
                js_invoke: js_invoke.clone(),
            }))
            .map_err(js_core_error)?;
    }

    let run = RunId::new(run_id).map_err(js_core_error)?;
    let mut ctx = RunContext::new(run, Some(u64::from(root_seed)));
    let phase = parse_phase(phase)?;
    let results = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, phase)
        .map_err(js_core_error)?;
    serde_json::to_string(&results).map_err(js_serde_error)
}
