//! Browser HPO adapter. Search, fold pruning, checkpoint validation and
//! selection remain in dag-ml-core; JavaScript owns operators and optimizer
//! state through synchronous callbacks.

use super::*;
use dag_ml_core::{
    ExternalDataPlanEnvelope, HostHpoCheckpoint, HostHpoInterruptedTrial, HostHpoProgress,
    HostHpoProposalSource, HostHpoResumeOptions, HostHpoSearchRequest, HostHpoSearchStatus,
    InMemoryDataProvider,
};

/// Validate a browser-persisted prepared terminal and seal interrupted trials
/// before resuming search. This is the same recovery contract as PyO3/C ABI.
#[wasm_bindgen]
pub fn recover_host_hpo_checkpoint_json(
    checkpoint_json: &str,
    prepared_json: &str,
    interrupted_json: &str,
) -> Result<String, JsValue> {
    let checkpoint: HostHpoCheckpoint =
        serde_json::from_str(checkpoint_json).map_err(js_serde_error)?;
    let prepared: Option<HostHpoCheckpoint> =
        serde_json::from_str(prepared_json).map_err(js_serde_error)?;
    let interrupted: Vec<HostHpoInterruptedTrial> =
        serde_json::from_str(interrupted_json).map_err(js_serde_error)?;
    let recovered = checkpoint
        .recover_interrupted_trials(prepared, interrupted)
        .map_err(js_core_error)?;
    serde_json::to_string(&recovered).map_err(js_serde_error)
}

fn call_optimizer(
    callback: &js_sys::Function,
    operation: &str,
    payload: serde_json::Value,
) -> CoreResult<serde_json::Value> {
    let payload = serde_json::to_string(&payload).map_err(CoreDagMlError::Serialization)?;
    let response = callback
        .call2(
            &JsValue::NULL,
            &JsValue::from_str(operation),
            &JsValue::from_str(&payload),
        )
        .map_err(|error| {
            CoreDagMlError::RuntimeValidation(format!(
                "JS HPO optimizer `{operation}` threw: {error:?}"
            ))
        })?;
    let json = response.as_string().ok_or_else(|| {
        CoreDagMlError::RuntimeValidation(format!(
            "JS HPO optimizer `{operation}` must return a JSON string"
        ))
    })?;
    serde_json::from_str(&json).map_err(CoreDagMlError::Serialization)
}

struct JsProposal {
    callback: js_sys::Function,
}

impl JsProposal {
    fn terminal(&self, operation: &str, payload: serde_json::Value) -> CoreResult<()> {
        let reply = call_optimizer(&self.callback, operation, payload)?;
        if reply.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
            Ok(())
        } else {
            Err(CoreDagMlError::RuntimeValidation(format!(
                "JS HPO optimizer `{operation}` must reply {{\"ok\":true}}"
            )))
        }
    }
}

impl HostHpoProposalSource for JsProposal {
    fn ask(&mut self, trial_index: u32) -> CoreResult<Option<BTreeMap<String, serde_json::Value>>> {
        self.ask_in_phase(trial_index, None)
    }

    fn ask_in_phase(
        &mut self,
        trial_index: u32,
        phase_index: Option<u32>,
    ) -> CoreResult<Option<BTreeMap<String, serde_json::Value>>> {
        let reply = call_optimizer(
            &self.callback,
            "ask",
            serde_json::json!({"trial_index": trial_index, "phase_index": phase_index}),
        )?;
        let params = reply.get("params").ok_or_else(|| {
            CoreDagMlError::RuntimeValidation("JS HPO ask reply needs `params`".into())
        })?;
        if params.is_null() {
            Ok(None)
        } else {
            serde_json::from_value(params.clone())
                .map(Some)
                .map_err(CoreDagMlError::Serialization)
        }
    }

    fn tell(&mut self, trial_index: u32, score: f64) -> CoreResult<()> {
        self.terminal(
            "tell",
            serde_json::json!({"trial_index":trial_index,"score":score}),
        )
    }

    fn report_intermediate(&mut self, trial_index: u32, step: u32, score: f64) -> CoreResult<bool> {
        let reply = call_optimizer(
            &self.callback,
            "report_intermediate",
            serde_json::json!({"trial_index":trial_index,"step":step,"score":score}),
        )?;
        reply
            .get("prune")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                CoreDagMlError::RuntimeValidation(
                    "JS HPO report_intermediate reply needs boolean `prune`".into(),
                )
            })
    }

    fn pruned(&mut self, trial_index: u32) -> CoreResult<()> {
        self.terminal("pruned", serde_json::json!({"trial_index":trial_index}))
    }

    fn fail(&mut self, trial_index: u32, error: &str) -> CoreResult<()> {
        self.terminal(
            "fail",
            serde_json::json!({"trial_index":trial_index,"error":error}),
        )
    }
}

struct JsProgress {
    callback: js_sys::Function,
}

impl HostHpoProgress for JsProgress {
    fn prepare_terminal(
        &mut self,
        checkpoint: &HostHpoCheckpoint,
        status: HostHpoSearchStatus,
    ) -> CoreResult<()> {
        let reply = call_optimizer(
            &self.callback,
            "prepare_terminal",
            serde_json::json!({"checkpoint":checkpoint,"status":status}),
        )?;
        if reply.get("ok").and_then(serde_json::Value::as_bool) == Some(true) {
            Ok(())
        } else {
            Err(CoreDagMlError::RuntimeValidation(
                "JS HPO prepare_terminal reply needs `ok: true`".into(),
            ))
        }
    }

    fn checkpoint(
        &mut self,
        checkpoint: &HostHpoCheckpoint,
        status: HostHpoSearchStatus,
    ) -> CoreResult<bool> {
        let reply = call_optimizer(
            &self.callback,
            "checkpoint",
            serde_json::json!({"checkpoint":checkpoint,"status":status}),
        )?;
        reply
            .get("continue")
            .and_then(serde_json::Value::as_bool)
            .ok_or_else(|| {
                CoreDagMlError::RuntimeValidation(
                    "JS HPO checkpoint reply needs boolean `continue`".into(),
                )
            })
    }
}

/// Run one-worker host HPO with native fold scoring, pruning and durable
/// checkpoint transitions. Both callbacks are synchronous: the controller is
/// `(controllerId, taskJson, exactSeed) => nodeResultJson`; the optimizer is
/// `(operation, payloadJson) => replyJson` (see README). The browser persists
/// its own optimizer state and native checkpoint before returning from each
/// `prepare_terminal`/`checkpoint` callback.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn host_hpo_search_json(
    execution_plan_json: &str,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    request_json: &str,
    checkpoint_json: Option<String>,
    js_invoke: &js_sys::Function,
    js_optimizer: &js_sys::Function,
) -> Result<String, JsValue> {
    let plan = ExecutionPlan::from_json(execution_plan_json).map_err(js_core_error)?;
    let trusted_manifests = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&plan, &trusted_manifests).map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        serde_json::from_str(data_envelope_json).map_err(js_serde_error)?;
    let mut options =
        HostHpoResumeOptions::from_envelope(&envelope, None).map_err(js_core_error)?;
    options.checkpoint = checkpoint_json
        .map(|json| serde_json::from_str(&json).map_err(js_serde_error))
        .transpose()?;
    let request: HostHpoSearchRequest =
        serde_json::from_str(request_json).map_err(js_serde_error)?;
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:wasm.hpo.provider").map_err(js_core_error)?,
        envelope,
    )
    .map_err(js_core_error)?;
    let mut controllers = RuntimeControllerRegistry::new();
    for manifest in plan.controller_manifests.values() {
        controllers
            .register(Box::new(JsRuntimeController {
                id: manifest.controller_id.clone(),
                js_invoke: js_invoke.clone(),
            }))
            .map_err(js_core_error)?;
    }
    let mut proposal = JsProposal {
        callback: js_optimizer.clone(),
    };
    let mut progress = JsProgress {
        callback: js_optimizer.clone(),
    };
    let outcome = SequentialScheduler
        .execute_resumable_host_hpo_search(
            &plan,
            &controllers,
            &provider,
            &request,
            &mut proposal,
            &options,
            &mut progress,
        )
        .map_err(js_core_error)?;
    serde_json::to_string(&outcome).map_err(js_serde_error)
}
