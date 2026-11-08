//! Actual allocation boundaries, using private state rather than production hooks.
use super::*;

fn fixture(name: &str) -> Value {
    let schema: Value = serde_json::from_str(include_str!(
        "../../../docs/contracts/realization_evidence.v1.schema.json"
    ))
    .unwrap();
    schema["x-conformance"]["valid"]
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v["id"] == name)
        .unwrap()
        .clone()
}

fn budget() -> Budget {
    Budget::new(Limits::default()).unwrap()
}

fn staged<T>(result: Result<T>, wanted: &str) {
    match result {
        Err(CaptureError::EvidenceBudgetExceeded {
            stage,
            limit_bytes,
            charged_bytes,
            requested_bytes,
        }) => {
            assert_eq!(stage, wanted);
            assert_eq!(charged_bytes + requested_bytes, limit_bytes + 1);
        }
        Err(other) => panic!("wrong refusal {other}"),
        Ok(_) => panic!("boundary was admitted: {wanted}"),
    }
}

#[test]
fn semantic_serialization_and_value_admission_boundaries() {
    let v = fixture("concrete_root_node_free_source_preserved");
    let event: Event = serde_json::from_value(v["events"][0].clone()).unwrap();
    let length = encoded_len(&event).unwrap();
    let serialization = 4 * length + 4096;
    let mut b = budget();
    b.limit = serialization - 1;
    staged(
        State::new().accept(&event, &mut b),
        "semantic serialization",
    );
    assert_eq!(b.used, 0);
    let raw = serde_json::to_string(&event).unwrap();
    b.limit = serialization + decoder_admission(&raw).unwrap().reservation_bytes - 1;
    staged(
        State::new().accept(&event, &mut b),
        "semantic value scratch",
    );
    assert_eq!(b.used, 0);
}

#[test]
fn actual_owned_index_and_retained_fact_boundaries() {
    let value = "a".repeat(8192);
    let mut b = budget();
    b.limit = value.len() - 1;
    staged(owned(&value, &mut b), "owned string");
    b.limit += 1;
    assert_eq!(owned(&value, &mut b).unwrap(), value);

    let mut index = Index::<Fact>::new();
    let mut b = budget();
    for i in 0..32 {
        index
            .insert(0, &format!("id:{i:04}"), Fact::empty(), &mut b)
            .unwrap();
    }
    let grow = 2 * index.nodes.capacity() * size_of::<IndexNode<Fact>>();
    b.limit = b.used + grow - 1;
    staged(
        index.insert(0, "id:next", Fact::empty(), &mut b),
        "reference index growth",
    );
    b.limit += 1;
    index.insert(0, "", Fact::empty(), &mut b).unwrap();

    let v = fixture("concrete_root_node_free_source_preserved");
    let e: Event = serde_json::from_value(v["events"][0].clone()).unwrap();
    let mut state = State::new();
    let mut b = budget();
    state.accept(&e, &mut b).unwrap();
    let data = Value::String(value);
    b.limit = b.used + retained_value_bound(&data).unwrap() - 1;
    staged(
        state.put("d", DECLARED, "fact", &data, &mut b),
        "retained fact",
    );
    assert!(state
        .scope("d")
        .unwrap()
        .facts
        .get(DECLARED, "fact")
        .is_none());
}

#[test]
fn seed_ancestry_scratch_boundary_is_requested_before_mutation() {
    let v = fixture("generated_zero_seed");
    let events: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let mut state = State::new();
    let mut b = budget();
    for e in &events[..6] {
        state.accept(e, &mut b).unwrap();
    }
    let reference = serde_json::json!({"scope_id":"s","id":"seed:effective"});
    b.limit = b.used + 8192 - 1;
    staged(
        state.seed_used(&reference, "p", "choice:a", &mut b),
        "seed ancestry scratch",
    );
    assert!(!state.fact("s", RESOLUTION, "seed:effective").unwrap().flag);
    b.limit += 1;
    state
        .seed_used(&reference, "p", "choice:a", &mut b)
        .unwrap();
    assert!(state.fact("s", RESOLUTION, "seed:effective").unwrap().flag);
}

#[test]
fn decoder_and_canonicalization_peak_boundaries() {
    let v = fixture("concrete_root_node_free_source_preserved");
    let expected = v["expected"].to_string();
    let p = decoder_admission(&expected).unwrap().reservation_bytes;
    let mut b = budget();
    b.limit = p;
    decode::<ExpectedWire>(&expected, &mut b, "expected context decode").unwrap();
    assert_eq!(b.peak, p);
    b.limit = p - 1;
    staged(
        decode::<ExpectedWire>(&expected, &mut b, "expected context decode"),
        "expected context decode",
    );

    let line = v["wire"].as_str().unwrap().lines().next().unwrap();
    let mut b = budget();
    b.limit = decoder_admission(line).unwrap().reservation_bytes - 1;
    staged(Decoder::new().feed(line, &mut b), "wire decoder");
    let WireRecord::Frame(f) = serde_json::from_str(line).unwrap() else {
        panic!("frame")
    };
    let mut b = budget();
    b.limit = preimage_reserve(&f).unwrap() - 1;
    staged(
        hash_record(&f, "frame_digest", FRAME_DOMAIN, &mut b),
        "wire canonicalization",
    );
    b.limit += 1;
    assert_eq!(
        hash_record(&f, "frame_digest", FRAME_DOMAIN, &mut b).unwrap(),
        f.frame_digest
    );
}

#[test]
fn logical_decoder_and_large_reassembly_have_real_boundaries() {
    let v = fixture("concrete_root_node_free_source_preserved");
    let mut decoder = Decoder::new();
    let raw = serde_json::to_vec(&v["events"][0]).unwrap();
    decoder.pending = raw;
    let mut b = budget();
    b.charge(decoder.pending.capacity(), "test preparation")
        .unwrap();
    let text = std::str::from_utf8(&decoder.pending).unwrap();
    b.limit = b.used + decoder_admission(text).unwrap().reservation_bytes - 1;
    staged(decoder.logical(&mut b), "logical decoder");

    // A large unfinished JSON whitespace prefix never enters the event parser.
    // Its next byte crosses an actual Vec growth boundary in feed, after hash.
    let mut decoder = Decoder::new();
    decoder.pending = vec![b' '; 262144];
    let mut b = budget();
    b.charge(decoder.pending.capacity(), "test preparation")
        .unwrap();
    let mut f = Frame {
        schema_version: 1,
        profile: PROFILE.into(),
        record_type: "frame".into(),
        capture_id: "capture:one".into(),
        sequence: 0,
        previous_digest: None,
        byte_length: 1,
        payload_hex: "20".into(),
        frame_digest: String::new(),
    };
    // This utility is used only for budget admission, never as a fault oracle.
    f.frame_digest = digest(&frame_preimage(&f).unwrap());
    let line = serde_json::to_string(&f).unwrap();
    let wire_reservation = decoder_admission(&line).unwrap().reservation_bytes;
    b.limit = b.used + wire_reservation + f.capture_id.len() + 524288 - 1;
    staged(decoder.feed(&line, &mut b), "reassembly growth");
}

#[test]
fn recorder_buffer_and_frame_buffer_boundaries() {
    staged(
        Recorder::new(
            io::sink(),
            "capture:one",
            Limits {
                memory_bytes: 8191,
                chunk_bytes: 8192,
            },
        ),
        "recorder buffer",
    );
    let v = fixture("concrete_root_node_free_source_preserved");
    let events: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let mut recorder = Recorder::new(io::sink(), "capture:one", Limits::default()).unwrap();
    for e in &events[..2] {
        recorder.record(e).unwrap();
    }
    recorder.budget.limit = recorder.budget.used + 10 * recorder.raw.len() + 20480 - 1;
    staged(recorder.emit_frame(), "frame buffers");
}

#[test]
fn declared_large_source_and_many_attempts_measure_admission() {
    let v = fixture("concrete_root_node_free_source_preserved");
    let mut e: Event = serde_json::from_value(v["events"][0].clone()).unwrap();
    if let Event::Declaration {
        declaration_text,
        source_digest,
        ..
    } = &mut e
    {
        *declaration_text = "opaque source ".repeat(2048);
        *source_digest = digest(declaration_text.as_bytes());
    }
    let mut r = Recorder::new(io::sink(), "capture:one", Limits::default()).unwrap();
    r.record(&e).unwrap();
    let declaration_peak = r.logical_peak_bytes();
    let attempts = fixture("failed_attempts_closed_carrier_unknown_ownership");
    let mut carrier: Event = serde_json::from_value(attempts["events"][1].clone()).unwrap();
    if let Event::CaptureProvenance {
        failed_attempts, ..
    } = &mut carrier
    {
        let one = failed_attempts[0].clone();
        *failed_attempts = vec![one; 8];
    }
    r.record(&carrier).unwrap();
    let admitted_peak = r.logical_peak_bytes();
    if let Event::CaptureProvenance {
        failed_attempts, ..
    } = &mut carrier
    {
        let one = failed_attempts[0].clone();
        *failed_attempts = vec![one; 64];
    }
    let error = r.record(&carrier).unwrap_err();
    assert_eq!(error.code(), "evidence_budget_exceeded");
    eprintln!(
        "declared corpus: source repetitions=2048, attempts=8/64; \
         logical peaks={declaration_peak}/{admitted_peak}; refusal={error:?}; RSS unmeasured"
    );
}

#[test]
fn owner_reference_facts_survive_proposals_and_release_with_search() {
    let v = fixture("adaptive_failed");
    let events: Vec<Event> = serde_json::from_value(v["events"].clone()).unwrap();
    let mut state = State::new();
    let mut b = budget();
    for e in &events[..13] {
        state.accept(e, &mut b).unwrap();
    }
    assert!(state.scope("p").is_err());
    let search = state.scope("s").unwrap();
    assert!(search
        .facts
        .get(OWNER_TRIAL, "747269616c3a6f6e65")
        .is_some());
    let retained = search.heap_bytes();
    let before = b.used;
    state.accept(&events[13], &mut b).unwrap();
    assert_eq!(before - b.used, retained);
    assert!(state.scope("s").is_err());
}
