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
fn run(
    ask_bytes: &[u8],
    bad_op: Option<&'static str>,
    parallel: u32,
    progressive: bool,
    pruning: bool,
) -> (DagMlStatusCode, String, Vec<String>) {
    let (status, error, events, _) = run_resumable(
        ask_bytes,
        bad_op,
        parallel,
        progressive,
        pruning,
        None,
        false,
    );
    (status, error, events)
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
#[test]
fn positive_ridge_and_report_refusal() {
    for parallel in [1, 2] {
        let (status, error, events) = run(&[], None, parallel, true, false);
        assert_eq!(status, DagMlStatusCode::OK, "{error}");
        assert!(events.iter().any(|e| e == "tell:2"));
        let (status, error, events) = run(&[], Some("report_intermediate"), parallel, true, false);
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
        assert!(error.contains("host HPO C feedback `report_intermediate` reply"));
        let report = events
            .iter()
            .rposition(|e| e.starts_with("report_intermediate:"))
            .unwrap();
        let prepare = events
            .iter()
            .position(|e| e == "prepare_terminal:failed")
            .unwrap();
        let fail = events.iter().position(|e| e.starts_with("fail:")).unwrap();
        let checkpoint = events
            .iter()
            .rposition(|e| e == "checkpoint:failed")
            .unwrap();
        assert!(report < prepare && prepare < fail && fail < checkpoint);
    }
}

#[test]
fn delivered_prepare_and_checkpoint_recover_and_resume() {
    for parallel in [1, 2] {
        for op in ["prepare_terminal", "checkpoint"] {
            let (status, error, _, checkpoints) =
                run_resumable(&[], Some(op), parallel, false, false, None, false);
            assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
            assert!(error.contains(&format!("host HPO C feedback `{op}` reply")));
            let delivered = serde_json::to_vec(&checkpoints.last().unwrap().1).unwrap();
            let resume = if op == "prepare_terminal" {
                let committed = serde_json::to_vec(
                    &checkpoints.iter().find(|x| x.0 == "checkpoint").unwrap().1,
                )
                .unwrap();
                let interrupted = b"[]";
                let mut out = DagMlOwnedBytes::default();
                let mut err = DagMlString::default();
                let status = unsafe {
                    dagml_host_hpo_checkpoint_recover_json(
                        committed.as_ptr(),
                        committed.len(),
                        delivered.as_ptr(),
                        delivered.len(),
                        interrupted.as_ptr(),
                        interrupted.len(),
                        &mut out,
                        &mut err,
                    )
                };
                assert_eq!(status, DagMlStatusCode::OK);
                let recovered = unsafe { slice::from_raw_parts(out.ptr, out.len).to_vec() };
                unsafe { dagml_owned_bytes_free(out) };
                recovered
            } else {
                delivered
            };
            let (status, error, _, _) =
                run_resumable(&[], None, parallel, false, false, Some(&resume), false);
            assert_eq!(status, DagMlStatusCode::OK, "{error}");
        }
    }
}
#[test]
fn strict_ask_and_pruned_refusal() {
    for parallel in [1, 2] {
        for v1 in [false, true] {
            for raw in [
                b"{\"offset\":1,\"offset\":2}".as_slice(),
                b"{\"offset\":1,\"o\\u0066fset\":2}",
                b"{\"offset\":18446744073709551616}",
                b"{\"offset\":1,\"\\u00e9\":1,\"e\\u0301\":2}",
                b"{\"offset\":1,\"x\":\"\\ud800\"}",
                b"null",
                b"[]",
                b"{",
                b"\xff",
            ] {
                let (status, error, events, _) =
                    run_resumable(raw, None, parallel, false, false, None, v1);
                assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
                assert!(error.contains("host HPO C ask parameters"));
                assert!(!events.iter().any(|e| e.starts_with("tell:")));
            }
            for raw in [
                b"{\"offset\":1,\"min\":-9223372036854775808,\"max\":18446744073709551615}"
                    .as_slice(),
                b"{\"offset\":1,\"n\":2e+19,\"z\":-0.0,\"tiny\":-2.5e-300}",
                b"{\"offset\":1,\"e\xcc\x81\":\"\xf0\x9f\x98\x80\"}",
            ] {
                let (status, error, _, _) =
                    run_resumable(raw, None, parallel, false, false, None, v1);
                assert_eq!(status, DagMlStatusCode::OK, "{error}");
            }
        }
        let (status, error, events) = run(&[], Some("pruned"), parallel, true, true);
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
        assert!(error.contains("host HPO C feedback `pruned` reply"));
        assert!(events.last().unwrap().starts_with("pruned:"));
    }
}

#[test]
fn v1_ask_ingress_sequential_and_parallel() {
    for parallel in [1, 2] {
        let (status, error, _, _) = run_resumable(&[], None, parallel, false, false, None, true);
        assert_eq!(status, DagMlStatusCode::OK, "{error}");
        let (status, error, _, _) = run_resumable(
            b"{\"offset\":1,\"offset\":2}",
            None,
            parallel,
            false,
            false,
            None,
            true,
        );
        assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
        assert!(error.contains("host HPO C ask parameters"));
    }
}

#[test]
fn recursion_band_is_labeled() {
    for (depth, selector) in [(127, "depth127"), (128, "depth128"), (129, "depth129")] {
        let raw = format!(
            "{{\"offset\":1,\"a\":{}1{}}}",
            "{\"a\":".repeat(depth - 2),
            "}".repeat(depth - 2)
        );
        for (v1, parallel) in [(false, 1), (false, 2), (true, 1), (true, 2)] {
            let (status, error, _, _) =
                run_resumable(raw.as_bytes(), None, parallel, false, false, None, v1);
            if depth < 129 {
                assert_eq!(status, DagMlStatusCode::OK, "{depth}: {error}");
            } else {
                assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
                assert!(error.contains("host HPO C ask parameters"));
            }
        }
        let (status, error, _) = run(&[], Some(selector), 1, false, false);
        if depth < 129 {
            assert_eq!(status, DagMlStatusCode::OK, "{depth}: {error}");
        } else {
            assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
            assert!(error.contains("host HPO C feedback `prepare_terminal` reply"));
        }
    }
}

#[test]
fn feedback_external_document_controls() {
    for parallel in [1, 2] {
        for selector in ["escaped", "nfc", "domain", "surrogate", "utf8", "shape"] {
            let (status, error, _) = run(&[], Some(selector), parallel, false, false);
            assert_eq!(
                status,
                DagMlStatusCode::VALIDATION_ERROR,
                "{selector}: {error}"
            );
            if selector != "shape" {
                assert!(
                    error.contains("host HPO C feedback `prepare_terminal` reply"),
                    "{selector}: {error}"
                );
            }
        }
        let (status, error, _) = run(&[], Some("legal"), parallel, false, false);
        assert_eq!(status, DagMlStatusCode::OK, "{error}");
    }
}
