//! Adopted integrity corrections. Literal wires are shared with Python/X1.
use super::*;
use dag_ml_core::realization_evidence::RefusalFlushStatus;
use sha2::{Digest, Sha256};
use std::io::BufWriter;

fn events(name: &str) -> Vec<Event> {
    serde_json::from_value(vector(name)["events"].clone()).unwrap()
}

#[test]
fn legacy_28_positive_60_negative_objects_remain_byte_identical() {
    let raw = include_str!("../../../../docs/contracts/realization_evidence.v1.schema.json");
    let pins: Value =
        serde_json::from_str(include_str!("../fixtures/realization_legacy_pins.json")).unwrap();
    for (section, count) in [("valid", 28), ("invalid", 60)] {
        let start = raw.find(&format!("\"{section}\": [")).unwrap();
        let mut at = start + raw[start..].find('[').unwrap() + 1;
        assert_eq!(pins[section].as_array().unwrap().len(), count);
        for pin in pins[section].as_array().unwrap() {
            at += raw[at..]
                .find(|c: char| !c.is_ascii_whitespace() && c != ',')
                .unwrap();
            let mut stream = serde_json::Deserializer::from_str(&raw[at..]).into_iter::<Value>();
            let entry = stream.next().unwrap().unwrap();
            let end = at + stream.byte_offset();
            assert_eq!(entry["id"], pin["id"]);
            assert_eq!(
                format!("{:x}", Sha256::digest(&raw.as_bytes()[at..end])),
                pin["raw_sha256"].as_str().unwrap(),
                "{}",
                entry["id"]
            );
            at = end;
        }
    }
}

#[test]
fn every_decodable_fixture_has_matching_event_and_payload_values() {
    let document = schema();
    for section in ["valid", "invalid", "parser_valid"] {
        for case in document["x-conformance"][section].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            if document["x-fixture-payload-exceptions"].get(id).is_some() {
                continue;
            }
            let expected = case
                .get("events")
                .unwrap_or(&document["x-fixture-events"][id]);
            assert!(expected.is_array(), "case={id}: missing event catalogue");
            let mut bytes = Vec::new();
            for line in case["wire"]
                .as_str()
                .unwrap()
                .split('\n')
                .filter(|s| !s.trim().is_empty())
            {
                let record: Value = serde_json::from_str(line).unwrap();
                if record["record_type"] == "frame" {
                    for pair in record["payload_hex"]
                        .as_str()
                        .unwrap()
                        .as_bytes()
                        .chunks_exact(2)
                    {
                        bytes.push(
                            u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap(),
                        );
                    }
                }
            }
            let decoded: Vec<Value> = std::str::from_utf8(&bytes)
                .unwrap()
                .split_terminator('\n')
                .map(|line| serde_json::from_str(line.strip_suffix('\r').unwrap_or(line)).unwrap())
                .collect();
            assert_eq!(
                &Value::Array(decoded),
                expected,
                "case={id}: event/wire identity"
            );
        }
    }
}

#[test]
fn nested_standalone_proposal_keeps_its_own_owner_namespace() {
    positive("integrity_nested_standalone_owner_ref");
    let v = vector("integrity_nested_standalone_owner_ref");
    let trials: Vec<_> = v["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["kind"] == "trial_visit")
        .collect();
    assert_eq!(trials.len(), 2);
    assert_eq!(
        trials[0]["owner_trial_ref_hex"],
        trials[1]["owner_trial_ref_hex"]
    );
    assert_ne!(trials[0]["scope_id"], trials[1]["scope_id"]);
}

#[test]
fn adopted_lifecycle_and_owner_cases_record_and_decode_exact_causes() {
    let doc = schema();
    let mut reached = std::collections::BTreeSet::new();
    for case in doc["x-conformance"]["invalid"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        if !id.starts_with("integrity_")
            || !case["events"].is_array()
            || case["refusal_at"] == "wire"
        {
            continue;
        }
        assert!(reached.insert(id));
        let input: Vec<Event> = serde_json::from_value(case["events"].clone()).unwrap();
        let mut r = Recorder::new(io::sink(), "capture:one", Limits::default()).unwrap();
        let error = if case["refusal_at"] == "finish" {
            for e in &input {
                r.record(e).unwrap();
            }
            r.finish().unwrap_err()
        } else {
            input
                .iter()
                .find_map(|e| r.record(e).err())
                .unwrap_or_else(|| {
                    panic!(
                        "case={id} expected_cause={}: record incorrectly accepted",
                        case["error"]
                    )
                })
        };
        assert_eq!(error.code(), case["error"].as_str().unwrap(), "{id}");
        assert_eq!(r.finish().unwrap_err().code(), "capture_terminal");
        assert_eq!(
            inspect_stream(case["wire"].as_str().unwrap(), Limits::default())
                .unwrap_err()
                .code(),
            error.code(),
            "case={id} expected_cause={}",
            case["error"]
        );
    }
    for id in [
        "integrity_trial_status_visit",
        "integrity_trial_status_resume",
        "integrity_trial_status_outcome_mode",
        "integrity_trial_status_outcome_initial",
        "integrity_provenance_carrier_event",
        "integrity_provenance_carrier_finish",
        "integrity_effective_seed_requested_null",
        "integrity_pending_visit_a_zero",
        "integrity_pending_trial",
        "integrity_open_children",
        "integrity_visit_disposition",
        "integrity_unvisited_evidence",
        "integrity_cleanup_observation",
        "integrity_sibling_owner_ref",
    ] {
        assert!(reached.contains(id), "missing lifecycle site {id}");
    }
}

#[test]
fn multiframe_capture_and_both_completion_count_checks_are_reachable() {
    let doc = schema();
    let cases = doc["x-conformance"]["invalid"].as_array().unwrap();
    for (id, cause) in [
        ("integrity_frame_capture_id", "capture_id"),
        ("integrity_completion_counts", "completion_counts"),
        (
            "integrity_completion_closed_scopes_digest",
            "completion_counts",
        ),
    ] {
        let case = cases.iter().find(|v| v["id"] == id).unwrap();
        let records: Vec<WireRecord> = case["wire"]
            .as_str()
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        if id == "integrity_frame_capture_id" {
            assert!(matches!(&records[1], WireRecord::Frame(f) if f.sequence == 1));
        }
        assert_eq!(
            inspect_stream(case["wire"].as_str().unwrap(), Limits::default())
                .unwrap_err()
                .code(),
            cause
        );
    }
    positive("integrity_multiframe");
}

#[derive(Default)]
struct ProbeSink {
    bytes: Vec<u8>,
    flushes: usize,
    writes: usize,
    fail_flush: bool,
    fail_write: bool,
}
impl Write for ProbeSink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        if self.fail_write {
            return Err(io::Error::other("write refusal"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.flushes += 1;
        if self.fail_flush {
            return Err(io::Error::other("flush-only refusal"));
        }
        Ok(())
    }
}

fn assert_prefix_only(bytes: &[u8], expected: &[Event]) {
    let wire = std::str::from_utf8(bytes).unwrap();
    let mut payload = Vec::new();
    for line in wire.lines() {
        let WireRecord::Frame(f) = serde_json::from_str(line).unwrap() else {
            panic!("a refusal must never synthesize a completion")
        };
        for pair in f.payload_hex.as_bytes().chunks_exact(2) {
            payload.push(u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap());
        }
    }
    let decoded: Vec<Event> = std::str::from_utf8(&payload)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(decoded, expected);
    assert_eq!(
        inspect_stream(wire, Limits::default()).unwrap_err().code(),
        "missing_completion"
    );
}

#[test]
fn semantic_refusal_flushes_buffered_prefix_at_most_once() {
    let es = events("concrete_root_node_free_source_preserved");
    let mut sink = ProbeSink::default();
    {
        let mut r = Recorder::new(&mut sink, "capture:one", Limits::default()).unwrap();
        for e in &es[..2] {
            r.record(e).unwrap();
        }
        let primary = r.record(&es[1]).unwrap_err();
        assert!(matches!(
            primary,
            CaptureError::Invalid("provenance_carrier")
        ));
        assert!(
            matches!(r.refusal_flush_status(), RefusalFlushStatus::Flushed),
            "case=accept_refused_buffered_prefix expected_cause=provenance_carrier"
        );
        assert_eq!(r.record(&es[2]).unwrap_err().code(), "capture_terminal");
        assert_eq!(r.finish().unwrap_err().code(), "capture_terminal");
    }
    assert_eq!(sink.flushes, 1);
    assert_prefix_only(&sink.bytes, &es[..2]);
}

#[test]
fn finish_refusal_flushes_only_previously_accepted_events() {
    let es = events("concrete_root_node_free_source_preserved");
    let accepted = [es[0].clone(), es.last().unwrap().clone()];
    let mut sink = ProbeSink::default();
    {
        let mut r = Recorder::new(&mut sink, "capture:one", Limits::default()).unwrap();
        for e in &accepted {
            r.record(e).unwrap();
        }
        assert_eq!(r.finish().unwrap_err().code(), "provenance_carrier");
        assert!(
            matches!(r.refusal_flush_status(), RefusalFlushStatus::Flushed),
            "case=finish_refused_buffered_prefix expected_cause=provenance_carrier"
        );
        assert_eq!(r.finish().unwrap_err().code(), "capture_terminal");
    }
    assert_eq!(sink.flushes, 1);
    assert_prefix_only(&sink.bytes, &accepted);
}

#[test]
fn refusal_primary_survives_secondary_flush_and_write_errors() {
    for fail_write in [false, true] {
        let es = events("concrete_root_node_free_source_preserved");
        let mut sink = ProbeSink {
            fail_write,
            fail_flush: !fail_write,
            ..ProbeSink::default()
        };
        {
            let mut r = Recorder::new(&mut sink, "capture:one", Limits::default()).unwrap();
            for e in &es[..2] {
                r.record(e).unwrap();
            }
            let primary = r.record(&es[1]).unwrap_err();
            assert!(matches!(
                primary,
                CaptureError::Invalid("provenance_carrier")
            ));
            assert!(matches!(
                r.refusal_flush_status(),
                RefusalFlushStatus::Failed(CaptureError::Sink(_))
            ));
            assert_eq!(r.finish().unwrap_err().code(), "capture_terminal");
        }
        assert_eq!(sink.flushes, usize::from(!fail_write));
        if fail_write {
            assert!(sink.bytes.is_empty());
            assert_eq!(sink.writes, 1);
        } else {
            assert_prefix_only(&sink.bytes, &es[..2]);
        }
    }
}

#[test]
fn budget_refusal_preserves_primary_when_diagnostic_flush_exceeds_budget() {
    let es = events("concrete_root_node_free_source_preserved");
    let mut baseline = Recorder::new(io::sink(), "capture:one", Limits::default()).unwrap();
    for e in &es[..2] {
        baseline.record(e).unwrap();
    }
    let prefix_peak = baseline.logical_peak_bytes();
    assert_eq!(
        baseline.record(&es[1]).unwrap_err().code(),
        "provenance_carrier"
    );
    assert!(matches!(
        baseline.refusal_flush_status(),
        RefusalFlushStatus::Flushed
    ));
    let flush_peak = baseline.logical_peak_bytes();
    assert!(
        flush_peak > prefix_peak,
        "the declared prefix must exercise shadow admission"
    );
    let limit = flush_peak - 1;
    let mut sink = ProbeSink::default();
    {
        let mut r = Recorder::new(
            &mut sink,
            "capture:one",
            Limits {
                memory_bytes: limit,
                ..Limits::default()
            },
        )
        .unwrap();
        for e in &es[..2] {
            r.record(e).unwrap();
        }
        let mut large = es[3].clone();
        if let Event::Choice { value_hex, .. } = &mut large {
            *value_hex = "00".repeat(limit);
        }
        match r.record(&large).unwrap_err() {
            CaptureError::EvidenceBudgetExceeded {
                stage, limit_bytes, ..
            } => {
                assert_eq!(stage, "semantic serialization");
                assert_eq!(limit_bytes, limit);
            }
            other => panic!("primary changed: {other}"),
        }
        assert!(matches!(
            r.refusal_flush_status(),
            RefusalFlushStatus::Failed(CaptureError::EvidenceBudgetExceeded { .. })
        ));
        assert_eq!(r.finish().unwrap_err().code(), "capture_terminal");
    }
    assert!(
        sink.bytes.is_empty(),
        "no prefix bytes may be claimed after failed shadow admission"
    );
    assert_eq!(sink.flushes, 0);
}

#[test]
fn operation_failed_flushes_through_a_real_bufwriter() {
    let es = events("concrete_root_node_free_source_preserved");
    let failure: Event = serde_json::from_value(serde_json::json!({
        "kind":"operation_failed","scope_id":"d","operation_id":"owner:ask",
        "stage":"ask","diagnostic_hex":"00","owner_ref_hex":null,
        "ownership":"unknown","consumption":"unknown"
    }))
    .unwrap();
    let mut sink = ProbeSink::default();
    {
        let buffer = BufWriter::with_capacity(65536, &mut sink);
        let mut r = Recorder::new(buffer, "capture:one", Limits::default()).unwrap();
        r.record(&es[0]).unwrap();
        r.record(&es[1]).unwrap();
        r.record(&failure).unwrap();
        // Do not call finish: its refusal flush would mask a missing operation flush.
    }
    assert_eq!(
        sink.flushes, 1,
        "case=operation_failed_buffered_prefix expected_cause=operation_sink_flush"
    );
    assert_prefix_only(&sink.bytes, &[es[0].clone(), es[1].clone(), failure]);
}

#[test]
fn sink_failure_during_push_operation_or_finish_never_retries() {
    let es = events("concrete_root_node_free_source_preserved");
    for mode in ["push", "operation", "finish"] {
        let mut sink = ProbeSink {
            fail_write: mode == "push",
            fail_flush: mode != "push",
            ..ProbeSink::default()
        };
        {
            let limits = Limits {
                chunk_bytes: if mode == "push" { 1 } else { 65536 },
                ..Limits::default()
            };
            let mut r = Recorder::new(&mut sink, "capture:one", limits).unwrap();
            let error = if mode == "push" {
                r.record(&es[0]).unwrap_err()
            } else if mode == "operation" {
                r.record(&es[0]).unwrap();
                let failure: Event = serde_json::from_value(serde_json::json!({
                    "kind":"operation_failed","scope_id":"d","operation_id":"owner:ask",
                    "stage":"ask","diagnostic_hex":"00","owner_ref_hex":null,
                    "ownership":"unknown","consumption":"unknown"
                }))
                .unwrap();
                r.record(&failure).unwrap_err()
            } else {
                for e in &es {
                    r.record(e).unwrap();
                }
                r.finish().unwrap_err()
            };
            assert!(matches!(error, CaptureError::Sink(_)), "{mode}");
            assert!(matches!(
                r.refusal_flush_status(),
                RefusalFlushStatus::NotAttempted
            ));
            assert_eq!(r.finish().unwrap_err().code(), "capture_terminal");
            assert_eq!(r.record(&es[0]).unwrap_err().code(), "capture_terminal");
        }
        assert_eq!(sink.flushes, usize::from(mode != "push"));
    }
}

fn scoped_facts(scopes: usize, limits: Limits) -> Result<usize, CaptureError> {
    let base = events("concrete_root_node_free_source_preserved");
    let mut r = Recorder::new(io::sink(), "capture:one", limits)?;
    r.record(&base[0])?;
    r.record(&base[1])?;
    for i in 0..scopes {
        let id = format!("p:{i:04}");
        for template in &base[2..7] {
            let mut e = template.clone();
            match &mut e {
                Event::ScopeOpen { scope_id, .. }
                | Event::Visit { scope_id, .. }
                | Event::Candidate { scope_id, .. }
                | Event::SearchClose { scope_id, .. } => *scope_id = id.clone(),
                Event::Choice {
                    scope_id,
                    value_hex,
                    ..
                } => {
                    *scope_id = id.clone();
                    *value_hex = "ab".repeat(8192);
                }
                _ => unreachable!(),
            }
            r.record(&e)?;
        }
    }
    r.record(base.last().unwrap())?;
    r.finish()?;
    Ok(r.logical_peak_bytes())
}

#[test]
fn releasing_closed_scope_facts_is_required_to_stay_under_limit() {
    let one = scoped_facts(
        1,
        Limits {
            chunk_bytes: 1024,
            ..Limits::default()
        },
    )
    .unwrap_or_else(|error| {
        panic!("case=repeated_scoped_fact_corpus expected_cause=scope_fact_release: {error:?}")
    });
    let limit = one + 128 * 4096; // accommodates index tombstones and overlapping growth
                                  // Even just the two retained string copies exceed this cap if not released.
    assert!(limit < 2 * 4 * 16384 * 128);
    scoped_facts(
        128,
        Limits {
            memory_bytes: limit,
            chunk_bytes: 1024,
        },
    )
    .unwrap_or_else(|error| {
        panic!("case=repeated_scoped_fact_corpus expected_cause=scope_fact_release: {error:?}")
    });
}

#[test]
fn recorder_peak_boundary_passes_at_p_and_refuses_at_p_minus_one() {
    let p = scoped_facts(
        1,
        Limits {
            chunk_bytes: 1024,
            ..Limits::default()
        },
    )
    .unwrap();
    assert_eq!(
        scoped_facts(
            1,
            Limits {
                memory_bytes: p,
                chunk_bytes: 1024,
            }
        )
        .unwrap(),
        p
    );
    match scoped_facts(
        1,
        Limits {
            memory_bytes: p - 1,
            chunk_bytes: 1024,
        },
    )
    .unwrap_err()
    {
        CaptureError::EvidenceBudgetExceeded {
            stage,
            charged_bytes,
            requested_bytes,
            ..
        } => {
            assert!(charged_bytes + requested_bytes > p - 1);
            // Final indexed identity in the overlapping shadow Choice acceptance.
            assert_eq!(stage, "owned string");
        }
        other => panic!("unexpected boundary refusal {other}"),
    }
}

#[test]
fn shadow_budget_failure_during_push_never_attempts_refusal_flush() {
    let es = events("concrete_root_node_free_source_preserved");
    let limits = Limits {
        chunk_bytes: 1,
        ..Limits::default()
    };
    let mut baseline = Recorder::new(io::sink(), "capture:one", limits).unwrap();
    baseline.record(&es[0]).unwrap();
    let p = baseline.logical_peak_bytes();
    let mut sink = ProbeSink::default();
    {
        let mut r = Recorder::new(
            &mut sink,
            "capture:one",
            Limits {
                memory_bytes: p - 1,
                ..limits
            },
        )
        .unwrap();
        assert!(matches!(
            r.record(&es[0]),
            Err(CaptureError::EvidenceBudgetExceeded { .. })
        ));
        assert!(matches!(
            r.refusal_flush_status(),
            RefusalFlushStatus::NotAttempted
        ));
        assert_eq!(r.finish().unwrap_err().code(), "capture_terminal");
    }
    assert_eq!(sink.flushes, 0);
    assert!(
        !sink.bytes.is_empty(),
        "failure must occur after push starts"
    );
    assert!(std::str::from_utf8(&sink.bytes)
        .unwrap()
        .lines()
        .all(|line| matches!(
            serde_json::from_str::<WireRecord>(line).unwrap(),
            WireRecord::Frame(_)
        )));
}

#[test]
fn verification_accessors_preserve_anchor_checks_and_success() {
    let v = vector("integrity_owner_encoded_fixed");
    let expected =
        decode_expected_context(Some(&v["expected"].to_string()), Limits::default()).unwrap();
    let verified =
        verify_stream(v["wire"].as_str().unwrap(), &expected, Limits::default()).unwrap();
    assert_eq!(
        verified.completion().completion_digest,
        expected.completion_digest()
    );
    let mut wrong = v["expected"].clone();
    wrong["completion_digest"] = Value::String("0".repeat(64));
    let wrong = decode_expected_context(Some(&wrong.to_string()), Limits::default()).unwrap();
    assert_eq!(
        verify_stream(v["wire"].as_str().unwrap(), &wrong, Limits::default())
            .unwrap_err()
            .code(),
        "expected_completion_mismatch"
    );
}

#[test]
fn owner_encoded_json_preimage_and_digest_are_literal() {
    use dag_ml_core::canonical::{parse_typed_json, tcv1_preimage};
    use dag_ml_core::realization_evidence::SeedValue;
    for v in schema()["x-owner-encoded-vectors"].as_array().unwrap() {
        let seed: SeedValue = serde_json::from_value(v["value"].clone()).unwrap();
        assert_eq!(
            serde_json::to_string(&seed).unwrap(),
            v["json"].as_str().unwrap()
        );
        let raw = tcv1_preimage(&parse_typed_json(v["json"].as_str().unwrap()).unwrap()).unwrap();
        let hex: String = raw.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, v["preimage_hex"].as_str().unwrap());
        assert_eq!(
            format!("{:x}", Sha256::digest(&raw)),
            v["sha256"].as_str().unwrap()
        );
    }
}
