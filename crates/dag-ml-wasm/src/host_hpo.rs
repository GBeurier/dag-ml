//! Browser HPO adapter. Search, fold pruning, checkpoint validation and
//! selection remain in dag-ml-core; JavaScript owns operators and optimizer
//! state through synchronous callbacks.

use super::*;
use dag_ml_core::{
    complete_host_hpo_worker_window, evaluate_host_hpo_worker_fold, evaluate_host_hpo_worker_task,
    host_hpo_worker_intermediate_score, host_hpo_worker_pruned_evidence,
    prepare_host_hpo_worker_window, validate_host_hpo_worker_fold_result, ExternalDataPlanEnvelope,
    HostHpoCheckpoint, HostHpoInterruptedTrial, HostHpoProgress, HostHpoProposalSource,
    HostHpoResumeOptions, HostHpoSearchRequest, HostHpoSearchStatus, HostHpoWorkerFoldResult,
    HostHpoWorkerResult, HostHpoWorkerTask, InMemoryDataProvider,
};
use wasm_bindgen_futures::JsFuture;

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

/// Evaluate one candidate inside its own Web Worker/WASM instance. The
/// browser's dispatcher sends the serialized `HostHpoWorkerTask` to a worker,
/// which invokes this function with its local synchronous controller callback.
#[wasm_bindgen]
pub fn host_hpo_evaluate_worker_task_json(
    task_json: &str,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    request_json: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let task: HostHpoWorkerTask = serde_json::from_str(task_json).map_err(js_serde_error)?;
    let request: HostHpoSearchRequest =
        serde_json::from_str(request_json).map_err(js_serde_error)?;
    let trusted_manifests = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&task.candidate_plan, &trusted_manifests)
        .map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        serde_json::from_str(data_envelope_json).map_err(js_serde_error)?;
    let data_fingerprint = HostHpoResumeOptions::from_envelope(&envelope, None)
        .map_err(js_core_error)?
        .data_fingerprint;
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:wasm.hpo.provider").map_err(js_core_error)?,
        envelope,
    )
    .map_err(js_core_error)?;
    let mut controllers = RuntimeControllerRegistry::new();
    for manifest in task.candidate_plan.controller_manifests.values() {
        controllers
            .register(Box::new(JsRuntimeController {
                id: manifest.controller_id.clone(),
                js_invoke: js_invoke.clone(),
            }))
            .map_err(js_core_error)?;
    }
    let evidence = evaluate_host_hpo_worker_task(&task, &request, &controllers, &provider)
        .map_err(js_core_error)?;
    serde_json::to_string(&HostHpoWorkerResult::Complete {
        evidence,
        data_fingerprint,
        fold_evidence: Vec::new(),
    })
    .map_err(js_serde_error)
}

/// Evaluate one FIT_CV fold only, then return to the browser coordinator for
/// its optimizer's prune decision before any later fold is dispatched.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn host_hpo_evaluate_worker_fold_json(
    task_json: &str,
    fold_index: u32,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    request_json: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let task: HostHpoWorkerTask = serde_json::from_str(task_json).map_err(js_serde_error)?;
    let request: HostHpoSearchRequest =
        serde_json::from_str(request_json).map_err(js_serde_error)?;
    let trusted_manifests = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&task.candidate_plan, &trusted_manifests)
        .map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        serde_json::from_str(data_envelope_json).map_err(js_serde_error)?;
    let data_fingerprint = HostHpoResumeOptions::from_envelope(&envelope, None)
        .map_err(js_core_error)?
        .data_fingerprint;
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:wasm.hpo.provider").map_err(js_core_error)?,
        envelope,
    )
    .map_err(js_core_error)?;
    let mut controllers = RuntimeControllerRegistry::new();
    for manifest in task.candidate_plan.controller_manifests.values() {
        controllers
            .register(Box::new(JsRuntimeController {
                id: manifest.controller_id.clone(),
                js_invoke: js_invoke.clone(),
            }))
            .map_err(js_core_error)?;
    }
    let result = evaluate_host_hpo_worker_fold(
        &task,
        &request,
        fold_index,
        &controllers,
        &provider,
        &data_fingerprint,
    )
    .map_err(js_core_error)?;
    serde_json::to_string(&result).map_err(js_serde_error)
}

async fn worker_json(promise: Result<js_sys::Promise, JsValue>) -> Result<String, String> {
    let promise = promise.map_err(|error| format!("worker dispatch threw: {error:?}"))?;
    let value = JsFuture::from(promise)
        .await
        .map_err(|error| format!("worker promise rejected: {error:?}"))?;
    value
        .as_string()
        .ok_or_else(|| "worker returned a non-string result".to_string())
}

/// The worker dispatcher receives tagged `{kind, task, fold_index?}` JSON in
/// pruning mode. Fold tasks are dispatched concurrently, but each candidate's
/// next fold waits for its native score to reach `report_intermediate`.
async fn dispatch_prunable_window(
    window: &dag_ml_core::HostHpoWorkerWindow,
    request: &HostHpoSearchRequest,
    data_fingerprint: &str,
    dispatch: &js_sys::Function,
    proposal: &mut JsProposal,
) -> Result<Vec<HostHpoWorkerResult>, JsValue> {
    let tasks = &window.tasks;
    let fold_count = tasks
        .first()
        .map(|task| {
            task.candidate_plan
                .fold_set
                .as_ref()
                .expect("validated FoldSet")
                .folds
                .len()
        })
        .unwrap_or(0);
    let mut active = (0..tasks.len()).collect::<Vec<_>>();
    let mut transcripts = vec![Vec::<HostHpoWorkerFoldResult>::new(); tasks.len()];
    let mut terminal = vec![None; tasks.len()];
    for fold_index in 0..fold_count {
        if active.is_empty() {
            break;
        }
        let pending = active
            .iter()
            .map(|&index| {
                let payload = serde_json::json!({
                    "kind": "fold",
                    "task": &tasks[index],
                    "fold_index": fold_index,
                });
                let payload = serde_json::to_string(&payload).map_err(js_serde_error)?;
                let promise = dispatch
                    .call1(&JsValue::NULL, &JsValue::from_str(&payload))
                    .map(|value| js_sys::Promise::resolve(&value));
                Ok::<_, JsValue>((index, promise))
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        let mut observed = Vec::with_capacity(pending.len());
        for (index, promise) in pending {
            match worker_json(promise).await {
                Ok(json) => {
                    let result: HostHpoWorkerFoldResult =
                        serde_json::from_str(&json).map_err(js_serde_error)?;
                    if result.fold_index as usize != fold_index {
                        return Err(JsValue::from_str("worker returned the wrong fold index"));
                    }
                    validate_host_hpo_worker_fold_result(
                        &tasks[index],
                        request,
                        data_fingerprint,
                        &result,
                    )
                    .map_err(js_core_error)?;
                    observed.push((index, Ok(result)));
                }
                Err(error) => observed.push((index, Err(error))),
            }
        }
        let mut next_active = Vec::new();
        for (index, result) in observed {
            let task = &tasks[index];
            match result {
                Err(error) => {
                    terminal[index] = Some(HostHpoWorkerResult::Failed {
                        trial_index: task.trial_index,
                        error,
                    });
                }
                Ok(result) => {
                    transcripts[index].push(result);
                    let intermediate = host_hpo_worker_intermediate_score(
                        task,
                        request,
                        data_fingerprint,
                        &transcripts[index],
                    )
                    .map_err(js_core_error)?;
                    if proposal
                        .report_intermediate(task.trial_index, fold_index as u32, intermediate)
                        .map_err(js_core_error)?
                    {
                        let evidence = host_hpo_worker_pruned_evidence(
                            task,
                            request,
                            data_fingerprint,
                            &transcripts[index],
                        )
                        .map_err(js_core_error)?;
                        terminal[index] = Some(HostHpoWorkerResult::Pruned {
                            evidence,
                            data_fingerprint: data_fingerprint.to_owned(),
                            fold_evidence: std::mem::take(&mut transcripts[index]),
                        });
                    } else {
                        next_active.push(index);
                    }
                }
            }
        }
        active = next_active;
    }
    let pending = active
        .iter()
        .map(|&index| {
            let payload = serde_json::json!({"kind": "complete", "task": &tasks[index]});
            let payload = serde_json::to_string(&payload).map_err(js_serde_error)?;
            let promise = dispatch
                .call1(&JsValue::NULL, &JsValue::from_str(&payload))
                .map(|value| js_sys::Promise::resolve(&value));
            Ok::<_, JsValue>((index, promise))
        })
        .collect::<Result<Vec<_>, JsValue>>()?;
    for (index, promise) in pending {
        let task = &tasks[index];
        let result = match worker_json(promise).await {
            Ok(json) => {
                let mut result: HostHpoWorkerResult =
                    serde_json::from_str(&json).map_err(js_serde_error)?;
                match &mut result {
                    HostHpoWorkerResult::Complete { fold_evidence, .. } => {
                        *fold_evidence = std::mem::take(&mut transcripts[index]);
                    }
                    _ => {
                        return Err(JsValue::from_str(
                            "full worker evaluation did not return complete evidence",
                        ));
                    }
                }
                result
            }
            Err(error) => HostHpoWorkerResult::Failed {
                trial_index: task.trial_index,
                error,
            },
        };
        terminal[index] = Some(result);
    }
    terminal
        .into_iter()
        .map(|result| result.ok_or_else(|| JsValue::from_str("worker trial has no terminal")))
        .collect()
}

/// Run a true browser worker window: dispatch every candidate before awaiting
/// any Promise, then reconcile results with native core in trial order. The
/// dispatcher has shape `(taskJson) => Promise<workerResultJson>`; each worker
/// owns a separate WASM instance and invokes `host_hpo_evaluate_worker_task_json`.
/// The existing synchronous `host_hpo_search_json` remains available.
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub async fn host_hpo_search_parallel_json(
    execution_plan_json: &str,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    request_json: &str,
    checkpoint_json: Option<String>,
    max_workers: usize,
    js_dispatch: &js_sys::Function,
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
    let mut proposal = JsProposal {
        callback: js_optimizer.clone(),
    };
    let mut progress = JsProgress {
        callback: js_optimizer.clone(),
    };
    let dispatch = js_dispatch.clone();
    let outcome = loop {
        let window =
            prepare_host_hpo_worker_window(&plan, &request, &options, &mut proposal, max_workers)
                .map_err(js_core_error)?;
        // All promises are created before the first await. Resolving each in
        // turn does not serialize worker execution.
        let results = if request.progressive_pruning {
            dispatch_prunable_window(
                &window,
                &request,
                &options.data_fingerprint,
                &dispatch,
                &mut proposal,
            )
            .await?
        } else {
            let pending = window
                .tasks
                .iter()
                .map(|task| {
                    let payload = serde_json::to_string(task).map_err(js_serde_error)?;
                    let response = dispatch.call1(&JsValue::NULL, &JsValue::from_str(&payload));
                    Ok::<_, JsValue>((
                        task.trial_index,
                        response.map(|value| js_sys::Promise::resolve(&value)),
                    ))
                })
                .collect::<Result<Vec<_>, JsValue>>()?;
            let mut results = Vec::with_capacity(pending.len());
            for (trial_index, promise) in pending {
                let result = worker_json(promise)
                    .await
                    .and_then(|json| serde_json::from_str(&json).map_err(|error| error.to_string()))
                    .unwrap_or_else(|error| HostHpoWorkerResult::Failed { trial_index, error });
                results.push(result);
            }
            results
        };
        let next = complete_host_hpo_worker_window(
            &plan,
            &request,
            &options,
            window,
            results,
            &mut proposal,
            &mut progress,
        )
        .map_err(js_core_error)?;
        if next.status != HostHpoSearchStatus::Running {
            break next;
        }
        options.checkpoint = next.checkpoint;
    };
    serde_json::to_string(&outcome).map_err(js_serde_error)
}
