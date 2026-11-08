//! Strict HPO callback ingress through the public C ABI and a small ridge host.
use dag_ml_capi::*;
use serde_json::{json, Value};
use std::{
    collections::HashSet,
    ffi::c_void,
    slice,
    sync::{Arc, Mutex},
};

const PLAN: &[u8] = include_bytes!("fixtures/host_hpo_ingress/plan.json");
const MANIFESTS: &[u8] = include_bytes!("fixtures/host_hpo_ingress/manifests.json");
const ENVELOPE: &[u8] = include_bytes!(
    "../../dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json"
);

#[derive(Default)]
struct Stats {
    events: Vec<String>,
    checkpoints: Vec<(String, Value)>,
    alloc: usize,
    release: usize,
    unknown_release: usize,
    live: HashSet<usize>,
    creates: usize,
    destroys: usize,
}
struct Host {
    stats: Arc<Mutex<Stats>>,
    ask: Vec<u8>,
    bad_op: Option<&'static str>,
    pruning: bool,
}
struct Candidate(Arc<Mutex<Stats>>);
#[derive(serde::Deserialize)]
struct TaskParams {
    offset: f64,
}
#[derive(serde::Deserialize)]
struct TaskNode {
    node_id: String,
    controller_version: String,
    params_fingerprint: String,
    params: TaskParams,
}
#[derive(serde::Deserialize)]
struct RidgeTask {
    node_plan: TaskNode,
    fold_id: String,
    variant_id: String,
    run_id: String,
    phase: Value,
    branch_path: Value,
    seed: Option<Value>,
}
#[derive(serde::Deserialize)]
struct ShallowFeedback {
    operation: String,
    status: Option<Value>,
    trial_index: Option<Value>,
}
unsafe fn give(out: *mut DagMlOwnedBytes, mut bytes: Vec<u8>, stats: &Mutex<Stats>) {
    let mut locked = stats.lock().unwrap();
    locked.alloc += 1;
    locked.live.insert(bytes.as_ptr() as usize);
    *out = DagMlOwnedBytes {
        ptr: bytes.as_mut_ptr(),
        len: bytes.len(),
        capacity: bytes.capacity(),
    };
    std::mem::forget(bytes);
}
unsafe extern "C" fn ask(
    user: *mut c_void,
    index: u32,
    _: i32,
    out: *mut DagMlOwnedBytes,
) -> DagMlStatusCode {
    let host = &*(user as *const Host);
    host.stats
        .lock()
        .unwrap()
        .events
        .push(format!("ask:{index}"));
    let reply = if host.ask.is_empty() {
        format!("{{\"offset\":{}}}", index + 1).into_bytes()
    } else {
        host.ask.clone()
    };
    give(out, reply, &host.stats);
    DagMlStatusCode::OK
}
unsafe extern "C" fn tell(user: *mut c_void, index: u32, score: f64) -> DagMlStatusCode {
    let host = &*(user as *const Host);
    host.stats
        .lock()
        .unwrap()
        .events
        .push(format!("tell:{index}"));
    let expected = [0.0, 0.7211102550927979, 0.9718253158075502];
    if host.ask.is_empty() {
        assert!((score - expected[index as usize]).abs() < 1e-12);
    }
    DagMlStatusCode::OK
}
unsafe extern "C" fn fail(user: *mut c_void, index: u32, _: DagMlBytesView) -> DagMlStatusCode {
    let host = &*(user as *const Host);
    host.stats
        .lock()
        .unwrap()
        .events
        .push(format!("fail:{index}"));
    DagMlStatusCode::OK
}
unsafe extern "C" fn release(user: *mut c_void, bytes: DagMlOwnedBytes) {
    let host = &*(user as *const Host);
    release_owned(bytes, &host.stats);
}
unsafe extern "C" fn create(user: *mut c_void, _: u32, out: *mut *mut c_void) -> DagMlStatusCode {
    let host = &*(user as *const Host);
    host.stats.lock().unwrap().creates += 1;
    *out = Box::into_raw(Box::new(Candidate(host.stats.clone()))).cast();
    DagMlStatusCode::OK
}
unsafe extern "C" fn destroy(user: *mut c_void) {
    let candidate = Box::from_raw(user as *mut Candidate);
    candidate.0.lock().unwrap().destroys += 1;
}
unsafe extern "C" fn invoke(
    user: *mut c_void,
    bytes: DagMlBytesView,
    out: *mut DagMlOwnedBytes,
) -> DagMlStatusCode {
    let candidate = &*(user as *const Candidate);
    let t: RidgeTask = serde_json::from_slice(slice::from_raw_parts(bytes.ptr, bytes.len)).unwrap();
    let fold = t.fold_id.as_str();
    let (train_x, validation_x, validation_id) = if fold == "fold:0" {
        (2.0, 1.0, "sample:1")
    } else {
        (1.0, 2.0, "sample:2")
    };
    let alpha = t.node_plan.params.offset - 1.0;
    let prediction = train_x * train_x / (train_x * train_x + alpha) * validation_x;
    let node = &t.node_plan.node_id;
    let result = json!({"node_id":node,"outputs":{},"predictions":[{"producer_node":node,"partition":"validation","fold_id":fold,"sample_ids":[validation_id],"values":[[prediction]],"target_names":["y"]}],
      "regression_targets":[{"level":"sample","unit_ids":[{"level":"sample","id":validation_id}],"values":[[validation_x]],"target_names":["y"]}],
      "lineage":{"record_id":format!("lineage:rust:{}:{}",t.variant_id,fold),"run_id":t.run_id,"node_id":node,"phase":t.phase,"controller_id":"controller:js-hpo-model","controller_version":t.node_plan.controller_version,"variant_id":t.variant_id,"fold_id":fold,"branch_path":t.branch_path,"input_lineage":[],"artifact_refs":[],"params_fingerprint":t.node_plan.params_fingerprint,"data_model_shape_fingerprint":null,"aggregation_policy_fingerprint":null,"seed":t.seed,"unsafe_flags":[],"metrics":{},"loss_attestations":[],"early_stopping_records":[]}});
    give(out, serde_json::to_vec(&result).unwrap(), &candidate.0);
    DagMlStatusCode::OK
}
unsafe extern "C" fn release_candidate(user: *mut c_void, bytes: DagMlOwnedBytes) {
    let candidate = &*(user as *const Candidate);
    release_owned(bytes, &candidate.0);
}
unsafe fn release_owned(bytes: DagMlOwnedBytes, stats: &Mutex<Stats>) {
    let mut locked = stats.lock().unwrap();
    if locked.live.remove(&(bytes.ptr as usize)) {
        locked.release += 1;
        drop(Vec::from_raw_parts(bytes.ptr, bytes.len, bytes.capacity));
    } else {
        locked.unknown_release += 1;
    }
}
unsafe extern "C" fn feedback(
    user: *mut c_void,
    bytes: DagMlBytesView,
    out: *mut DagMlOwnedBytes,
) -> DagMlStatusCode {
    let host = &*(user as *const Host);
    let data = slice::from_raw_parts(bytes.ptr, bytes.len);
    let event: Value = if host.ask.len() > 400 {
        let shallow: ShallowFeedback = serde_json::from_slice(data).unwrap();
        json!({"operation":shallow.operation,"status":shallow.status,"trial_index":shallow.trial_index})
    } else {
        serde_json::from_slice(data).unwrap()
    };
    let op = event["operation"].as_str().unwrap();
    if let Some(checkpoint) = event.get("checkpoint") {
        host.stats
            .lock()
            .unwrap()
            .checkpoints
            .push((op.to_owned(), checkpoint.clone()));
    }
    host.stats.lock().unwrap().events.push(format!(
        "{}:{}",
        op,
        event.get("status").and_then(Value::as_str).unwrap_or("")
    ));
    let should_refuse =
        host.bad_op == Some(op) && (op != "report_intermediate" || event["trial_index"] == 1);
    let depth = if op == "prepare_terminal" {
        match host.bad_op {
            Some("depth127") => Some(127),
            Some("depth128") => Some(128),
            Some("depth129") => Some(129),
            _ => None,
        }
    } else {
        None
    };
    let prepare_override = if op == "prepare_terminal" {
        match host.bad_op {
            Some("escaped") => Some(b"{\"ok\":true,\"o\\u006b\":true}".as_slice()),
            Some("nfc") => Some(b"{\"ok\":true,\"\\u00e9\":1,\"e\\u0301\":2}".as_slice()),
            Some("domain") => Some(b"{\"ok\":true,\"n\":18446744073709551616}".as_slice()),
            Some("surrogate") => Some(b"{\"ok\":true,\"x\":\"\\ud800\"}".as_slice()),
            Some("utf8") => Some(b"\xff".as_slice()),
            Some("shape") => Some(b"[]".as_slice()),
            Some("legal") => Some(b"{\"ok\":true,\"min\":-9223372036854775808,\"max\":18446744073709551615,\"f\":2e+19,\"e\\u0301\":\"\\ud83d\\ude00\"}".as_slice()),
            _ => None,
        }
    } else {
        None
    };
    let reply = if let Some(reply) = prepare_override {
        reply.to_vec()
    } else if let Some(depth) = depth {
        format!(
            "{{\"ok\":true,\"x\":{}1{}}}",
            "[".repeat(depth - 2),
            "]".repeat(depth - 2)
        )
        .into_bytes()
    } else if should_refuse {
        if op == "report_intermediate" {
            b"{\"prune\":true,\"prune\":false}".to_vec()
        } else {
            b"{\"ok\":false,\"ok\":true}".to_vec()
        }
    } else if op == "report_intermediate" {
        if host.pruning {
            b"{\"prune\":true}".to_vec()
        } else {
            b"{\"prune\":false}".to_vec()
        }
    } else {
        b"{\"ok\":true}".to_vec()
    };
    give(out, reply, &host.stats);
    DagMlStatusCode::OK
}
fn run_resumable(
    ask_bytes: &[u8],
    bad_op: Option<&'static str>,
    parallel: u32,
    progressive: bool,
    pruning: bool,
    resume: Option<&[u8]>,
    v1: bool,
) -> (DagMlStatusCode, String, Vec<String>, Vec<(String, Value)>) {
    let mut host = Host {
        stats: Arc::new(Mutex::new(Stats::default())),
        ask: ask_bytes.to_vec(),
        bad_op,
        pruning,
    };
    let user = (&mut host as *mut Host).cast();
    let callbacks = DagMlHostHpoCallbacks {
        abi_version: 1,
        user_data: user,
        ask: Some(ask),
        tell: Some(tell),
        fail: Some(fail),
        release_proposal_bytes: Some(release),
        create_candidate: Some(create),
        invoke_candidate: Some(invoke),
        release_candidate_bytes: Some(release_candidate),
        destroy_candidate: Some(destroy),
    };
    let feedback = DagMlHostHpoFeedbackCallbacks {
        abi_version: 1,
        user_data: user,
        invoke: Some(feedback),
        release_bytes: Some(release),
    };
    let request = serde_json::to_vec(&json!({"target_node":"model:m","trial_budget":3,"metric":"rmse","direction":"minimize","optimizer_descriptor":{"owner":"rust-test"},"progressive_pruning":progressive})).unwrap();
    let mut out = DagMlOwnedBytes::default();
    let mut error = DagMlString::default();
    let status = unsafe {
        if v1 {
            dagml_host_hpo_search_json(
                PLAN.as_ptr(),
                PLAN.len(),
                MANIFESTS.as_ptr(),
                MANIFESTS.len(),
                ENVELOPE.as_ptr(),
                ENVELOPE.len(),
                request.as_ptr(),
                request.len(),
                callbacks,
                parallel,
                &mut out,
                &mut error,
            )
        } else {
            dagml_host_hpo_search_json_v2(
                PLAN.as_ptr(),
                PLAN.len(),
                MANIFESTS.as_ptr(),
                MANIFESTS.len(),
                ENVELOPE.as_ptr(),
                ENVELOPE.len(),
                request.as_ptr(),
                request.len(),
                resume.map_or(std::ptr::null(), |v| v.as_ptr()),
                resume.map_or(0, <[u8]>::len),
                callbacks,
                feedback,
                parallel,
                &mut out,
                &mut error,
            )
        }
    };
    let text = unsafe {
        if error.ptr.is_null() {
            String::new()
        } else {
            let s = String::from_utf8_lossy(slice::from_raw_parts(error.ptr.cast(), error.len))
                .into_owned();
            dagml_string_free(error);
            s
        }
    };
    if !out.ptr.is_null() {
        unsafe { dagml_owned_bytes_free(out) };
    }
    let stats = host.stats.lock().unwrap();
    assert_eq!(stats.alloc, stats.release);
    assert_eq!(stats.unknown_release, 0);
    assert!(stats.live.is_empty());
    assert_eq!(stats.creates, stats.destroys);
    (
        status,
        text,
        stats.events.clone(),
        stats.checkpoints.clone(),
    )
}

type RequestFn = unsafe extern "C" fn(
    *const u8,
    usize,
    *mut DagMlOwnedBytes,
    *mut DagMlString,
) -> DagMlStatusCode;

fn check(function: RequestFn, request: &Value, valid: bool) {
    let bytes = serde_json::to_vec(request).unwrap();
    let mut out = DagMlOwnedBytes::default();
    let mut error = DagMlString::default();
    let status = unsafe { function(bytes.as_ptr(), bytes.len(), &mut out, &mut error) };
    let message = if error.ptr.is_null() {
        String::new()
    } else {
        String::from_utf8_lossy(unsafe { slice::from_raw_parts(error.ptr.cast(), error.len) })
            .into_owned()
    };
    if valid {
        assert_eq!(status, DagMlStatusCode::OK, "{message}");
    } else {
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR, "{message}");
        assert!(message.contains("unknown field"), "{message}");
        assert!(message.contains("serialization"), "{message}");
        assert!(out.ptr.is_null());
    }
    unsafe {
        dagml_owned_bytes_free(out);
        dagml_string_free(error);
    }
}

#[test]
fn stacking_reports_refuse_unknowns_without_closing_metric_names() {
    let report = json!({"producer_node":"model:a", "partition":"validation", "fold_id":"fold:0", "level":"sample", "row_count":1, "target_width":1, "metrics":{"rmse":4.0, "custom":1.0}});
    let fold = json!({"producer_node":"model:a", "fold_ids":["fold:0"], "metric":"rmse", "reports":[report.clone()]});
    let producers =
        json!({"producer_nodes":["model:a"], "select":"best", "metric":"rmse", "reports":[report]});
    for (function, request) in [
        (dagml_select_stacking_fold_json as RequestFn, fold.clone()),
        (dagml_stacking_fold_weights_json as RequestFn, fold),
        (dagml_select_stacking_producers_json as RequestFn, producers),
    ] {
        check(function, &request, true);
        let mut changed = request;
        changed["reports"][0]["unknown_probe"] = json!(true);
        check(function, &changed, false);
    }
}

#[test]
fn report_to_candidate_refuses_unknowns() {
    let mut report = json!({"producer_node":"model:a", "partition":"validation", "fold_id":"fold:0", "level":"sample", "row_count":1, "target_width":1, "metrics":{"rmse":4.0}});
    for valid in [true, false] {
        if !valid {
            report["unknown_probe"] = json!(true);
        }
        let bytes = serde_json::to_vec(&report).unwrap();
        let candidate = b"candidate:one";
        let mut out = DagMlOwnedBytes::default();
        let mut error = DagMlString::default();
        let status = unsafe {
            dagml_regression_report_candidate_score_json(
                bytes.as_ptr(),
                bytes.len(),
                DagMlBytesView {
                    ptr: candidate.as_ptr(),
                    len: candidate.len(),
                },
                &mut out,
                &mut error,
            )
        };
        if valid {
            assert_eq!(status, DagMlStatusCode::OK);
        } else {
            let message = String::from_utf8_lossy(unsafe {
                slice::from_raw_parts(error.ptr.cast(), error.len)
            });
            assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
            assert!(message.contains("unknown field"));
        }
        unsafe {
            dagml_owned_bytes_free(out);
            dagml_string_free(error);
        }
    }
}

#[test]
fn owner_checkpoint_recovery_and_resume_refuse_nested_unknowns() {
    for pruning in [false, true] {
        let (status, error, _, checkpoints) =
            run_resumable(b"", None, 1, pruning, pruning, None, false);
        assert_eq!(status, DagMlStatusCode::OK, "{error}");
        let checkpoint = checkpoints
            .iter()
            .rev()
            .find(|(op, value)| {
                op == "checkpoint"
                    && value["trials"]
                        .as_array()
                        .is_some_and(|trials| !trials.is_empty())
            })
            .unwrap()
            .1
            .clone();
        for field in ["evidence", "scores", "report"] {
            let mut changed = checkpoint.clone();
            let evidence = &mut changed["trials"][0]["evidence"];
            let target = match field {
                "evidence" => evidence,
                "scores" => &mut evidence["scores"],
                _ => &mut evidence["scores"]["reports"][0],
            };
            target["unknown_probe"] = json!(true);
            let bytes = serde_json::to_vec(&changed).unwrap();
            let mut out = DagMlOwnedBytes::default();
            let mut error = DagMlString::default();
            let interrupted = b"[]";
            let status = unsafe {
                dagml_host_hpo_checkpoint_recover_json(
                    bytes.as_ptr(),
                    bytes.len(),
                    std::ptr::null(),
                    0,
                    interrupted.as_ptr(),
                    interrupted.len(),
                    &mut out,
                    &mut error,
                )
            };
            let message = String::from_utf8_lossy(unsafe {
                slice::from_raw_parts(error.ptr.cast(), error.len)
            })
            .into_owned();
            assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
            assert!(
                message.contains("serialization error") && message.contains("unknown field"),
                "{message}"
            );
            unsafe {
                dagml_owned_bytes_free(out);
                dagml_string_free(error);
            }
            let (status, message, events, _) =
                run_resumable(b"", None, 1, pruning, pruning, Some(&bytes), false);
            assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
            assert!(message.contains("unknown field"), "{message}");
            assert!(events.is_empty(), "{events:?}");
        }
    }
}
