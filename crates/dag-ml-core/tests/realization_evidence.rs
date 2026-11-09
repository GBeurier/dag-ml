//! Contract-only regression source. AUTHORED, NOT EXECUTED in source-only grant.
use dag_ml_core::realization_evidence::{
    completion_preimage, decode_expected_context, decoder_admission, frame_preimage,
    inspect_stream, verify_stream, CaptureError, Event, Limits, Recorder, WireRecord,
};
use serde_json::Value;
use std::io::{self, Write};
fn schema() -> Value {
    serde_json::from_str(include_str!(
        "fixtures/package/realization_evidence.schema.json"
    ))
    .unwrap()
}
fn vector(name: &str) -> Value {
    schema()["x-conformance"]["valid"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == name)
        .unwrap()
        .clone()
}
fn capture(v: &Value, chunk: usize) -> String {
    let events: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let before = serde_json::to_value(&events).unwrap();
    let mut bytes = Vec::new();
    {
        let mut r = Recorder::new(
            &mut bytes,
            "capture:one",
            Limits {
                chunk_bytes: chunk,
                ..Limits::default()
            },
        )
        .unwrap();
        for e in &events {
            r.record(e).unwrap();
        }
        let finished = r.finish().unwrap();
        assert_eq!(
            finished.completion().completion_digest,
            finished.expected_commitment().completion_digest()
        );
        assert!(r.logical_peak_bytes() <= Limits::default().memory_bytes);
    }
    assert_eq!(serde_json::to_value(events).unwrap(), before);
    String::from_utf8(bytes).unwrap()
}
fn positive(name: &str) {
    let v = vector(name);
    let expected =
        decode_expected_context(Some(&v["expected"].to_string()), Limits::default()).unwrap();
    let wire = capture(&v, v["chunk_bytes"].as_u64().unwrap() as usize);
    assert_eq!(wire, v["wire"].as_str().unwrap());
    verify_stream(&wire, &expected, Limits::default()).unwrap();
}
#[test]
fn all_shared_literal_recorder_bytes_and_digests() {
    for v in schema()["x-conformance"]["valid"].as_array().unwrap() {
        positive(v["id"].as_str().unwrap());
    }
}
#[test]
fn all_literal_preimages_are_independent_expected_bytes() {
    for v in schema()["x-conformance"]["valid"].as_array().unwrap() {
        assert_eq!(
            v["wire"].as_str().unwrap().lines().count(),
            v["preimages_hex"].as_array().unwrap().len(),
            "{}",
            v["id"]
        );
        for (line, expected) in v["wire"]
            .as_str()
            .unwrap()
            .lines()
            .zip(v["preimages_hex"].as_array().unwrap())
        {
            let r: WireRecord = serde_json::from_str(line).unwrap();
            let raw = match r {
                WireRecord::Frame(f) => frame_preimage(&f).unwrap(),
                WireRecord::Completion(c) => completion_preimage(&c).unwrap(),
            };
            let actual: String = raw.iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(actual, expected.as_str().unwrap());
        }
    }
}
#[test]
fn every_negative_checks_its_specific_cause() {
    for case in schema()["x-conformance"]["invalid"].as_array().unwrap() {
        assert_negative_case(case);
    }
}
fn assert_negative_case(case: &Value) {
    let id = case["id"].as_str().unwrap();
    let cause = case["error"].as_str().unwrap();
    let wire = case["wire"].as_str().unwrap();
    if case["structural_pass"] == true {
        inspect_stream(wire, Limits::default()).unwrap_or_else(|error| {
            panic!("case={id} expected_cause={cause}: structural control failed: {error:?}")
        });
    }
    let result = if case.get("expected").is_some() {
        let expected =
            decode_expected_context(Some(&case["expected"].to_string()), Limits::default())
                .unwrap_or_else(|error| {
                    panic!("case={id} expected_cause={cause}: expected decode failed: {error:?}")
                });
        verify_stream(wire, &expected, Limits::default()).map(|_| ())
    } else {
        inspect_stream(wire, Limits::default()).map(|_| ())
    };
    let error = result
        .err()
        .unwrap_or_else(|| panic!("case={id} expected_cause={cause}: incorrectly accepted"));
    assert_eq!(error.code(), cause, "case={id} expected_cause={cause}");
}
fn assert_negative_fixture(id: &str) {
    let document = schema();
    let cases: Vec<_> = document["x-conformance"]["invalid"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case["id"].as_str() == Some(id))
        .collect();
    assert_eq!(cases.len(), 1, "case={id}: expected one fixture");
    assert_negative_case(cases[0]);
}
#[test]
fn lf_crlf_and_final_eof_have_explicit_controls() {
    for case in schema()["x-conformance"]["parser_valid"]
        .as_array()
        .unwrap()
    {
        let expected =
            decode_expected_context(Some(&case["expected"].to_string()), Limits::default())
                .unwrap();
        verify_stream(case["wire"].as_str().unwrap(), &expected, Limits::default()).unwrap();
    }
}
#[test]
fn same_parameter_two_nodes_is_not_flattened() {
    positive("same_parameter_different_located_nodes");
    let v = vector("same_parameter_different_located_nodes");
    let c: Vec<_> = v["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["kind"] == "choice")
        .collect();
    assert_eq!(c.len(), 2);
    assert_ne!(c[0]["node_id"], c[1]["node_id"]);
    assert_eq!(c[0]["source_path_hex"], c[1]["source_path_hex"]);
}
#[test]
fn preproposal_failed_pruned_cancelled_are_retained() {
    for name in ["adaptive_failed", "adaptive_pruned", "adaptive_cancelled"] {
        positive(name);
        let v = vector(name);
        assert_eq!(
            v["events"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["kind"] == "candidate")
                .count(),
            0
        );
    }
}
#[test]
fn restored_stochastic_records_make_no_new_draw_claim() {
    positive("restored_stochastic_completed_no_new_draw");
    let v = vector("restored_stochastic_completed_no_new_draw");
    for e in v["events"].as_array().unwrap() {
        if e["kind"] == "effective_seed_resolved" || e["kind"] == "choice" {
            assert_eq!(e["provenance"]["observation"], "restored");
            assert!(!e["provenance"]["checkpoint_ref"].is_null());
        }
    }
}
#[test]
fn zero_trials_absent_and_zero_remain_distinct() {
    positive("zero_trials_requested_zero_no_resolution");
    positive("zero_trials_requested_absent");
    positive("cancel_before_draw");
    let a = vector("zero_trials_requested_zero_no_resolution");
    let b = vector("zero_trials_requested_absent");
    assert_ne!(a["wire"], b["wire"]);
}
#[test]
fn generated_and_derived_seed_controls() {
    positive("generated_zero_seed");
    positive("derived_full_u64_unused_requested_parent");
    positive("derived_effective_ancestor_consumption");
    positive("explicit_stochastic_use_outside_choice");
}
#[test]
fn local_shared_nested_clone_mapping() {
    positive("local_shared_nested_clones");
}
#[test]
fn retry_can_materialize_after_earlier_failure() {
    positive("retry_candidate_after_failed_attempt");
}
#[test]
fn equal_owner_id_in_different_scopes_is_not_deduplicated() {
    positive("same_owner_id_distinct_scopes");
}
#[test]
fn failed_attempts_have_a_closed_wire_carrier() {
    positive("failed_attempts_closed_carrier_unknown_ownership");
}
#[test]
fn source_spelling_and_utf8_fragments_are_preserved() {
    positive("opaque_numeric_spelling_and_separators");
    positive("fragmented_utf8_exact_nfc_and_nfd_bytes");
}
#[test]
fn missing_expected_is_a_decode_refusal() {
    assert_eq!(
        decode_expected_context(None, Limits::default())
            .err()
            .unwrap()
            .code(),
        "expected_required"
    );
    assert_eq!(
        decode_expected_context(Some("null"), Limits::default())
            .err()
            .unwrap()
            .code(),
        "expected_required"
    );
    assert_eq!(
        decode_expected_context(Some("{}"), Limits::default())
            .err()
            .unwrap()
            .code(),
        "expected_shape"
    );
}
#[test]
fn contextual_verification_rejects_fully_rechained_catalogues() {
    for c in schema()["x-conformance"]["invalid"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["structural_pass"] == true)
    {
        inspect_stream(c["wire"].as_str().unwrap(), Limits::default()).unwrap();
        let expected =
            decode_expected_context(Some(&c["expected"].to_string()), Limits::default()).unwrap();
        assert_eq!(
            verify_stream(c["wire"].as_str().unwrap(), &expected, Limits::default())
                .err()
                .unwrap()
                .code(),
            "expected_completion_mismatch"
        );
    }
}
struct BrokenSink;
impl Write for BrokenSink {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("synthetic sink refusal"))
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[test]
fn failed_sink_never_returns_expected_anchor() {
    let v = vector("concrete_root_node_free_source_preserved");
    let es: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let mut r = Recorder::new(
        BrokenSink,
        "capture:one",
        Limits {
            chunk_bytes: 1,
            ..Limits::default()
        },
    )
    .unwrap();
    assert!(matches!(r.record(&es[0]), Err(CaptureError::Sink(_))));
    assert_eq!(r.finish().err().unwrap().code(), "capture_terminal");
}
#[test]
fn budget_refusal_precedes_emission_and_is_staged() {
    let v = vector("concrete_root_node_free_source_preserved");
    let es: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let mut bytes = Vec::new();
    {
        let mut r = Recorder::new(
            &mut bytes,
            "capture:one",
            Limits {
                memory_bytes: 4096,
                chunk_bytes: 1,
            },
        )
        .unwrap();
        match r.record(&es[0]).err().unwrap() {
            CaptureError::EvidenceBudgetExceeded { stage, .. } => {
                assert_eq!(stage, "semantic serialization")
            }
            e => panic!("unexpected {e}"),
        };
        assert!(r.finish().is_err());
    }
    assert!(bytes.is_empty());
}
#[test]
fn dense_tokens_get_precharged_before_generic_decode() {
    let dense = format!("[{}0]", "0,".repeat(100000));
    let admission = decoder_admission(&dense).unwrap();
    assert!(admission.nodes >= 100001);
    assert!(admission.reservation_bytes > dense.len() * 32);
    let error = decode_expected_context(
        Some(&dense),
        Limits {
            memory_bytes: 4096,
            chunk_bytes: 1,
        },
    )
    .err()
    .unwrap();
    match error {
        CaptureError::EvidenceBudgetExceeded { stage, .. } => {
            assert_eq!(stage, "expected context decode")
        }
        e => panic!("unexpected {e}"),
    };
}
#[test]
fn released_scope_indices_do_not_retain_prior_source_text() {
    let v = vector("concrete_root_node_free_source_preserved");
    let events: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let mut r = Recorder::new(Vec::new(), "capture:one", Limits::default()).unwrap();
    for e in &events {
        r.record(e).unwrap();
    }
    let finished = r.finish().unwrap();
    let wire = String::from_utf8(capture(&v, 65536).into_bytes()).unwrap();
    verify_stream(&wire, finished.expected_commitment(), Limits::default()).unwrap();
    assert!(r.lookup_comparisons() < 10000);
}

#[test]
fn operation_failure_reaches_only_diagnostic_frames() {
    let v = vector("zero_trials_requested_zero_no_resolution");
    let events: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let mut bytes = Vec::new();
    {
        let mut recorder = Recorder::new(&mut bytes, "capture:one", Limits::default()).unwrap();
        for e in &events[..4] {
            recorder.record(e).unwrap();
        }
        let failure:Event=serde_json::from_value(serde_json::json!({"kind":"operation_failed","scope_id":"s","operation_id":"actual:ask","stage":"parameter_resolution","diagnostic_hex":"6572726f72","owner_ref_hex":null,"ownership":"unknown","consumption":"unknown"})).unwrap();
        recorder.record(&failure).unwrap();
        assert_eq!(recorder.finish().err().unwrap().code(), "incomplete_error");
    }
    assert!(!bytes.is_empty());
    let wire = std::str::from_utf8(&bytes).unwrap();
    for line in wire.lines() {
        assert!(matches!(
            serde_json::from_str::<WireRecord>(line).unwrap(),
            WireRecord::Frame(_)
        ));
    }
    assert_eq!(
        inspect_stream(wire, Limits::default())
            .err()
            .unwrap()
            .code(),
        "missing_completion"
    );
}

#[test]
fn expected_context_closed_shape_and_grammar() {
    let base = vector("concrete_root_node_free_source_preserved");
    for (field, value) in [
        ("schema_version", serde_json::json!(2)),
        ("profile", serde_json::json!("wrong-profile")),
        ("capture_id", serde_json::json!("name/invalid")),
        ("completion_digest", serde_json::json!("F".repeat(64))),
    ] {
        let mut expected = base["expected"].clone();
        expected[field] = value;
        assert_eq!(
            decode_expected_context(Some(&expected.to_string()), Limits::default())
                .err()
                .unwrap()
                .code(),
            "expected_invalid"
        );
    }
    let mut extra = base["expected"].clone();
    extra["extra"] = Value::Null;
    assert_eq!(
        decode_expected_context(Some(&extra.to_string()), Limits::default())
            .err()
            .unwrap()
            .code(),
        "expected_shape"
    );
}

#[test]
fn closed_scope_reference_work_is_bounded_without_a_clock() {
    let v = vector("concrete_root_node_free_source_preserved");
    let events: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let mut recorder = Recorder::new(io::sink(), "capture:one", Limits::default()).unwrap();
    recorder.record(&events[0]).unwrap();
    recorder.record(&events[1]).unwrap();
    // Actual proposal identities, no model or engine work. Closed scope tombstones
    // remain in a balanced index; facts/source buffers are released per scope.
    for i in 0..1024 {
        let id = format!("scope:{i:04}");
        let open:Event=serde_json::from_value(serde_json::json!({"kind":"scope_open","scope_id":id,"scope_kind":"proposal","parent_scope_id":"d","declaration_ref":{"scope_id":"d","source_digest":v["events"][0]["source_digest"]}})).unwrap();
        recorder.record(&open).unwrap();
        let close:Event=serde_json::from_value(serde_json::json!({"kind":"search_close","scope_id":id,"reason":"exhausted","unvisited":false,"basis":"enumeration_complete","evidence":{"availability":"known","value_hex":"00"}})).unwrap();
        recorder.record(&close).unwrap();
    }
    recorder.record(events.last().unwrap()).unwrap();
    recorder.finish().unwrap();
    // >500000 comparisons for linear search on these 1024 scopes; generous
    // deterministic bound covers both recorder and same-Limits shadow decoder.
    assert!(recorder.lookup_comparisons() < 400000);
    assert!(recorder.logical_peak_bytes() <= Limits::default().memory_bytes);
}

const REVIEWER_W_CASE_IDS: &[&str] = &[
    "w1_seed_conflicting_two_resolutions_one_declaration",
    "w2_shared_clone_records_local_choice",
    "w3_restored_trial_on_new_visit",
    "w3_new_trial_on_restored_visit",
    "w3_restored_candidate_on_new_visit",
    "w3_restored_choice_in_new_visit",
    "w4_exhausted_known_empty_evidence",
    "w4_restored_empty_checkpoint_ref",
    "w4_empty_owner_trial_ref",
    "w4_known_empty_stage",
    "w4_known_empty_reason",
    "w3_checkpoint_binding_mismatch",
    "w1_duplicate_resolution_same_value",
    "w1_declaration_outlives_first_proposal",
];
const OWNER_REFERENCE_CASE_IDS: &[&str] = &[
    "w4_confirmed_attempt_empty_ref",
    "w4_confirmed_operation_empty_ref",
    "w4_prefix_empty_owner_ref",
];

fn assert_event_cases_exactly_once(ids: &[&str]) {
    let document = schema();
    let cases = document["x-conformance"]["invalid"].as_array().unwrap();
    let mut reached = std::collections::BTreeSet::new();
    for id in ids {
        assert!(reached.insert(*id), "duplicate intended case: {id}");
        let matches: Vec<_> = cases
            .iter()
            .filter(|case| case["id"].as_str() == Some(*id))
            .collect();
        assert_eq!(matches.len(), 1, "expected one fixture: {id}");
        let case = matches[0];
        assert!(case["events"].is_array(), "missing event input: {id}");
        let events: Vec<Event> = serde_json::from_value(case["events"].clone()).unwrap();
        let mut recorder = Recorder::new(io::sink(), "capture:one", Limits::default()).unwrap();
        let error = events
            .iter()
            .find_map(|event| recorder.record(event).err())
            .expect("event case must refuse before completion");
        assert_eq!(error.code(), case["error"].as_str().unwrap(), "{id}");
        assert_eq!(recorder.finish().err().unwrap().code(), "capture_terminal");
        assert_eq!(
            inspect_stream(case["wire"].as_str().unwrap(), Limits::default())
                .err()
                .unwrap()
                .code(),
            case["error"].as_str().unwrap(),
            "{id}"
        );
    }
    assert_eq!(reached.len(), ids.len());
}

#[test]
fn reviewer_w1_w4_record_and_decode_refuse_exact_causes() {
    assert_eq!(REVIEWER_W_CASE_IDS.len(), 14);
    assert_event_cases_exactly_once(REVIEWER_W_CASE_IDS);
}

#[test]
fn confirmed_owner_reference_cases_refuse_before_completion() {
    assert_event_cases_exactly_once(OWNER_REFERENCE_CASE_IDS);
    positive("confirmed_attempt_nonempty_ref_null_prefix");
    positive("trusted_prefix_nonempty_ref");
}

#[test]
fn independent_declarations_and_unrelated_origins_are_supported() {
    positive("independent_scoped_seed_resolutions");
    positive("unrelated_new_and_restored_proposals");
    positive("unrelated_new_restored_same_proposal");
}

#[path = "realization_evidence/integrity.rs"]
mod integrity;

#[path = "realization_evidence/mutant_cases.rs"]
mod mutant_cases;
