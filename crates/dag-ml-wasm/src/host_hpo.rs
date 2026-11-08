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

fn decode_host_hpo_argument<T: serde::de::DeserializeOwned + serde::Serialize>(
    raw: &str,
    label: &str,
) -> Result<T, JsValue> {
    deserialize_external_contract(raw, label, CoreDagMlError::RuntimeValidation)
        .map_err(|error| match error {
            CoreDagMlError::Serialization(error) => {
                CoreDagMlError::RuntimeValidation(format!("{label}: {error}"))
            }
            other => other,
        })
        .map_err(js_core_error)
}

/// Validate a browser-persisted prepared terminal and seal interrupted trials
/// before resuming search. This is the same recovery contract as PyO3/C ABI.
#[wasm_bindgen]
pub fn recover_host_hpo_checkpoint_json(
    checkpoint_json: &js_sys::JsString,
    prepared_json: &js_sys::JsString,
    interrupted_json: &js_sys::JsString,
) -> Result<String, JsValue> {
    let checkpoint_json = crate::js_text::text_argument(checkpoint_json, "checkpoint_json")?;
    let prepared_json = crate::js_text::text_argument(prepared_json, "prepared_json")?;
    let interrupted_json = crate::js_text::text_argument(interrupted_json, "interrupted_json")?;
    recover_host_hpo_checkpoint_json_impl(&checkpoint_json, &prepared_json, &interrupted_json)
}

fn recover_host_hpo_checkpoint_json_impl(
    checkpoint_json: &str,
    prepared_json: &str,
    interrupted_json: &str,
) -> Result<String, JsValue> {
    let checkpoint: HostHpoCheckpoint =
        decode_host_hpo_argument(checkpoint_json, "host HPO recovery checkpoint")?;
    let prepared: Option<HostHpoCheckpoint> =
        decode_host_hpo_argument(prepared_json, "host HPO recovery prepared checkpoint")?;
    let interrupted: Vec<HostHpoInterruptedTrial> =
        decode_host_hpo_argument(interrupted_json, "host HPO recovery interrupted trials")?;
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
                "JS HPO optimizer `{operation}` threw: {}",
                crate::js_text::js_failure_text(&error)
            ))
        })?;
    let json = callback_utf16_string(&response)
        .map_err(|()| {
            CoreDagMlError::RuntimeValidation(format!(
                "JS HPO optimizer `{operation}` reply is not valid UTF-16"
            ))
        })?
        .ok_or_else(|| {
            CoreDagMlError::RuntimeValidation(format!(
                "JS HPO optimizer `{operation}` must return a JSON string"
            ))
        })?;
    deserialize_external_contract::<serde_json::Value, _>(
        &json,
        &format!("JS HPO optimizer `{operation}` reply"),
        CoreDagMlError::RuntimeValidation,
    )
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
    execution_plan_json: &js_sys::JsString,
    trusted_controller_manifests_json: &js_sys::JsString,
    data_envelope_json: &js_sys::JsString,
    request_json: &js_sys::JsString,
    checkpoint_json: Option<js_sys::JsString>,
    js_invoke: &js_sys::Function,
    js_optimizer: &js_sys::Function,
) -> Result<String, JsValue> {
    let execution_plan_json =
        crate::js_text::text_argument(execution_plan_json, "execution_plan_json")?;
    let trusted_controller_manifests_json = crate::js_text::text_argument(
        trusted_controller_manifests_json,
        "trusted_controller_manifests_json",
    )?;
    let data_envelope_json =
        crate::js_text::text_argument(data_envelope_json, "data_envelope_json")?;
    let request_json = crate::js_text::text_argument(request_json, "request_json")?;
    let checkpoint_json =
        crate::js_text::optional_text_argument(checkpoint_json, "checkpoint_json")?;
    host_hpo_search_json_impl(
        &execution_plan_json,
        &trusted_controller_manifests_json,
        &data_envelope_json,
        &request_json,
        checkpoint_json.as_deref(),
        js_invoke,
        js_optimizer,
    )
}

#[allow(clippy::too_many_arguments)]
fn host_hpo_search_json_impl(
    execution_plan_json: &str,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    request_json: &str,
    checkpoint_json: Option<&str>,
    js_invoke: &js_sys::Function,
    js_optimizer: &js_sys::Function,
) -> Result<String, JsValue> {
    let plan = ExecutionPlan::from_json(execution_plan_json).map_err(js_core_error)?;
    let trusted_manifests = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&plan, &trusted_manifests).map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        decode_host_hpo_argument(data_envelope_json, "host HPO envelope")?;
    let mut options =
        HostHpoResumeOptions::from_envelope(&envelope, None).map_err(js_core_error)?;
    options.checkpoint = checkpoint_json
        .map(|json| decode_host_hpo_argument(json, "host HPO search checkpoint"))
        .transpose()?;
    let request: HostHpoSearchRequest = decode_host_hpo_argument(request_json, "host HPO request")?;
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
    task_json: &js_sys::JsString,
    trusted_controller_manifests_json: &js_sys::JsString,
    data_envelope_json: &js_sys::JsString,
    request_json: &js_sys::JsString,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let task_json = crate::js_text::text_argument(task_json, "task_json")?;
    let trusted_controller_manifests_json = crate::js_text::text_argument(
        trusted_controller_manifests_json,
        "trusted_controller_manifests_json",
    )?;
    let data_envelope_json =
        crate::js_text::text_argument(data_envelope_json, "data_envelope_json")?;
    let request_json = crate::js_text::text_argument(request_json, "request_json")?;
    host_hpo_evaluate_worker_task_json_impl(
        &task_json,
        &trusted_controller_manifests_json,
        &data_envelope_json,
        &request_json,
        js_invoke,
    )
}

fn host_hpo_evaluate_worker_task_json_impl(
    task_json: &str,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    request_json: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let task: HostHpoWorkerTask = decode_host_hpo_argument(task_json, "host HPO worker task")?;
    let request: HostHpoSearchRequest = decode_host_hpo_argument(request_json, "host HPO request")?;
    let trusted_manifests = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&task.candidate_plan, &trusted_manifests)
        .map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        decode_host_hpo_argument(data_envelope_json, "host HPO envelope")?;
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
    task_json: &js_sys::JsString,
    fold_index: u32,
    trusted_controller_manifests_json: &js_sys::JsString,
    data_envelope_json: &js_sys::JsString,
    request_json: &js_sys::JsString,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let task_json = crate::js_text::text_argument(task_json, "task_json")?;
    let trusted_controller_manifests_json = crate::js_text::text_argument(
        trusted_controller_manifests_json,
        "trusted_controller_manifests_json",
    )?;
    let data_envelope_json =
        crate::js_text::text_argument(data_envelope_json, "data_envelope_json")?;
    let request_json = crate::js_text::text_argument(request_json, "request_json")?;
    host_hpo_evaluate_worker_fold_json_impl(
        &task_json,
        fold_index,
        &trusted_controller_manifests_json,
        &data_envelope_json,
        &request_json,
        js_invoke,
    )
}

fn host_hpo_evaluate_worker_fold_json_impl(
    task_json: &str,
    fold_index: u32,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    request_json: &str,
    js_invoke: &js_sys::Function,
) -> Result<String, JsValue> {
    let task: HostHpoWorkerTask = decode_host_hpo_argument(task_json, "host HPO worker task")?;
    let request: HostHpoSearchRequest = decode_host_hpo_argument(request_json, "host HPO request")?;
    let trusted_manifests = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&task.candidate_plan, &trusted_manifests)
        .map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        decode_host_hpo_argument(data_envelope_json, "host HPO envelope")?;
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

// Adopt each dispatch result through a fresh native Promise and attach its
// handlers before awaiting any other worker in the window.
fn observe_worker_dispatch(response: Result<JsValue, JsValue>) -> Result<JsFuture, JsValue> {
    response.map(|value| {
        let promise = js_sys::Promise::new(&mut |resolve, _reject| {
            let _ = resolve.call1(&JsValue::UNDEFINED, &value);
        });
        JsFuture::from(promise)
    })
}

async fn worker_json(future: Result<JsFuture, JsValue>) -> Result<String, String> {
    let future = future.map_err(|error| {
        format!(
            "worker dispatch threw: {}",
            crate::js_text::js_failure_text(&error)
        )
    })?;
    let value = future.await.map_err(|error| {
        format!(
            "worker promise rejected: {}",
            crate::js_text::js_failure_text(&error)
        )
    })?;
    callback_utf16_string(&value)
        .map_err(|()| "worker returned a result with invalid UTF-16".to_string())?
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
                .map(|set| set.folds.len())
                .ok_or_else(|| JsValue::from_str("prunable worker task requires a FoldSet"))
        })
        .transpose()?
        .unwrap_or(0);
    if tasks.iter().any(|task| {
        task.candidate_plan
            .fold_set
            .as_ref()
            .is_none_or(|set| set.folds.len() != fold_count)
    }) {
        return Err(JsValue::from_str(
            "prunable worker tasks must share the fold count",
        ));
    }
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
                let promise = observe_worker_dispatch(
                    dispatch.call1(&JsValue::NULL, &JsValue::from_str(&payload)),
                );
                Ok::<_, JsValue>((index, promise))
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        let mut observed = Vec::with_capacity(pending.len());
        for (index, promise) in pending {
            match worker_json(promise).await {
                Ok(json) => {
                    let parsed = (|| -> Result<HostHpoWorkerFoldResult, String> {
                        let result: HostHpoWorkerFoldResult =
                            dag_ml_core::deserialize_external_contract(
                                &json,
                                "worker fold",
                                CoreDagMlError::RuntimeValidation,
                            )
                            .map_err(|error| error.to_string())?;
                        if result.fold_index as usize != fold_index {
                            return Err("worker returned the wrong fold index".into());
                        }
                        validate_host_hpo_worker_fold_result(
                            &tasks[index],
                            request,
                            data_fingerprint,
                            &result,
                        )
                        .map_err(|error| error.to_string())?;
                        Ok(result)
                    })();
                    observed.push((index, parsed));
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
            let promise = observe_worker_dispatch(
                dispatch.call1(&JsValue::NULL, &JsValue::from_str(&payload)),
            );
            Ok::<_, JsValue>((index, promise))
        })
        .collect::<Result<Vec<_>, JsValue>>()?;
    for (index, promise) in pending {
        let task = &tasks[index];
        let result = match worker_json(promise).await {
            Ok(json) => match deserialize_external_contract::<HostHpoWorkerResult, _>(
                &json,
                "worker complete result",
                CoreDagMlError::RuntimeValidation,
            ) {
                Ok(HostHpoWorkerResult::Complete {
                    evidence,
                    data_fingerprint,
                    ..
                }) => HostHpoWorkerResult::Complete {
                    evidence,
                    data_fingerprint,
                    fold_evidence: std::mem::take(&mut transcripts[index]),
                },
                Ok(_) => HostHpoWorkerResult::Failed {
                    trial_index: task.trial_index,
                    error: "full worker evaluation did not return complete evidence".into(),
                },
                Err(error) => HostHpoWorkerResult::Failed {
                    trial_index: task.trial_index,
                    error: error.to_string(),
                },
            },
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
    execution_plan_json: &js_sys::JsString,
    trusted_controller_manifests_json: &js_sys::JsString,
    data_envelope_json: &js_sys::JsString,
    request_json: &js_sys::JsString,
    checkpoint_json: Option<js_sys::JsString>,
    max_workers: usize,
    js_dispatch: &js_sys::Function,
    js_optimizer: &js_sys::Function,
) -> Result<String, JsValue> {
    let execution_plan_json =
        crate::js_text::text_argument(execution_plan_json, "execution_plan_json")?;
    let trusted_controller_manifests_json = crate::js_text::text_argument(
        trusted_controller_manifests_json,
        "trusted_controller_manifests_json",
    )?;
    let data_envelope_json =
        crate::js_text::text_argument(data_envelope_json, "data_envelope_json")?;
    let request_json = crate::js_text::text_argument(request_json, "request_json")?;
    let checkpoint_json =
        crate::js_text::optional_text_argument(checkpoint_json, "checkpoint_json")?;
    host_hpo_search_parallel_json_impl(
        &execution_plan_json,
        &trusted_controller_manifests_json,
        &data_envelope_json,
        &request_json,
        checkpoint_json.as_deref(),
        max_workers,
        js_dispatch,
        js_optimizer,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn host_hpo_search_parallel_json_impl(
    execution_plan_json: &str,
    trusted_controller_manifests_json: &str,
    data_envelope_json: &str,
    request_json: &str,
    checkpoint_json: Option<&str>,
    max_workers: usize,
    js_dispatch: &js_sys::Function,
    js_optimizer: &js_sys::Function,
) -> Result<String, JsValue> {
    let plan = ExecutionPlan::from_json(execution_plan_json).map_err(js_core_error)?;
    let trusted_manifests = controller_registry_from_json(trusted_controller_manifests_json)?;
    validate_runtime_controller_manifests(&plan, &trusted_manifests).map_err(js_core_error)?;
    let envelope: ExternalDataPlanEnvelope =
        decode_host_hpo_argument(data_envelope_json, "host HPO envelope")?;
    let mut options =
        HostHpoResumeOptions::from_envelope(&envelope, None).map_err(js_core_error)?;
    options.checkpoint = checkpoint_json
        .map(|json| decode_host_hpo_argument(json, "host HPO parallel search checkpoint"))
        .transpose()?;
    let request: HostHpoSearchRequest = decode_host_hpo_argument(request_json, "host HPO request")?;
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
                    Ok::<_, JsValue>((task.trial_index, observe_worker_dispatch(response)))
                })
                .collect::<Result<Vec<_>, JsValue>>()?;
            let mut results = Vec::with_capacity(pending.len());
            for (trial_index, promise) in pending {
                let result = worker_json(promise)
                    .await
                    .and_then(|json| {
                        deserialize_external_contract::<HostHpoWorkerResult, _>(
                            &json,
                            "worker result",
                            CoreDagMlError::RuntimeValidation,
                        )
                        .map_err(|error| error.to_string())
                    })
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
