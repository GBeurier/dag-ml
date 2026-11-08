//! Plan-free capture contracts: literal adversarial vectors, the closed recorder
//! verdicts on them, and the cross-language pair golden.

use dag_ml_core::realization_evidence::{Event, FailedAttempt, Limits, Recorder};
use dag_ml_core::*;
use serde_json::{json, Value};

const DECLARATION: &str = r#"{"search":{"n_components":[1,2,3]}}"#;
const DECLARATION_DIGEST: &str = "a46fbd129a3f4265ddb6515a22f39f72d14de5b71faecbbce7b3f938cfe206e2";
/// Owner digests and pairs below come from the SDK TCV1 encoder
/// (`nirs4all/pipeline/dagml/training_contracts.py`) and were checked against the
/// repository encoder, so these literals compare Rust with both.
const OWNER_STATE_DOCUMENT: &str = r##"{"trial_rows":[{"owner_ref":"a0","status":"failed"}],"sampler":{"kind":"random","rng_state_hex":"726e673a31"},"pending_params":{},"config":{"sampler":"random","seed":42}}"##;
const OWNER_STATE: &str = "c0d0edf62656295dde8db953170c12734322f9c3e9d69cd4c4a3918a361e7209";
const PAIR_FINGERPRINT: &str = "397ad25b75e48cb82a22115bee658b819900fa2c6fad9488a8bc8b549145d1ea";
const CHECKPOINT_FINGERPRINT: &str =
    "1281ad05e7b6f093a6f0a3d4d775c0f2191fe261941bcbecd951c927fe8ad727";
/// A sealed Failed prefix with integer and binary64 parameters.
const PUBLICATION: &str = r##"{"checkpoint":{"schema_version":2,"checkpoint":{"schema_version":1,"binding":{"objective_fingerprint":"1111111111111111111111111111111111111111111111111111111111111111","graph_fingerprint":"2222222222222222222222222222222222222222222222222222222222222222","controller_fingerprint":"3333333333333333333333333333333333333333333333333333333333333333","campaign_fingerprint":"4444444444444444444444444444444444444444444444444444444444444444","fold_set_fingerprint":"5555555555555555555555555555555555555555555555555555555555555555","data_fingerprint":"fixture:data"},"trials":[{"state":"failed","trial_index":0,"params":{"alpha":0.1,"n_components":2,"scale":1.0,"shift":-2.5},"variant_id":"host_hpo:trial:0000000000","error":"runtime validation failed: fit failed"}],"fingerprint":"b86b63e9ba93bfbdb40dcc40116f6b1e9000bf5bc99e07b14f1a7c687e4d4812"},"source_digest":"a46fbd129a3f4265ddb6515a22f39f72d14de5b71faecbbce7b3f938cfe206e2","attempts":[{"trial_index":0,"capture_id":"capture:first","scope_id":"declaration:search","search_scope_id":"search:capture","owner_ref_hex":"a0","owner_evidence_hex":"6f776e65722d747269616c3a30","phase_index":null,"seed_declarations":[{"seed_id":"seed:sampler","requested":null}],"seed_resolutions":[{"seed_id":"seed:sampler","effective":{"kind":"u64","value":42},"resolution":"generated","parent":null,"policy":{"id":"policy:sampler","version":"1","context_hex":"706f6c6963792d636f6e74657874"},"first_actual_use_id":"choice:n_components"}],"seed_uses":[],"located_choices":[{"choice_id":"choice:n_components","parameter":"n_components","node_id":"model:pls","source_path_hex":["6e5f636f6d706f6e656e7473"],"value_hex":"32","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:alpha","parameter":"alpha","node_id":"model:pls","source_path_hex":["616c706861"],"value_hex":"302e31","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:scale","parameter":"scale","node_id":"model:pls","source_path_hex":["7363616c65"],"value_hex":"312e30","semantics":"deterministic","seed_id":null},{"choice_id":"choice:shift","parameter":"shift","node_id":"model:pls","source_path_hex":["7368696674"],"value_hex":"2d322e35","semantics":"deterministic","seed_id":null}],"candidate":{"owner_variant_id":"host_hpo:trial:0000000000","owner_fingerprint":"9999999999999999999999999999999999999999999999999999999999999999","active_node_ids":["model:pls","transform:snv"],"parameters_hex":"7b22616c706861223a302e312c226e5f636f6d706f6e656e7473223a322c227363616c65223a312e302c227368696674223a2d322e357d"},"failure_stage":"evaluation","causal_diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a20666974206661696c6564","terminal_capture_id":"capture:first","terminal_scope_id":"declaration:search","owner_observation":{"owner_ref_hex":"a0","ownership":"confirmed","status":"failed","cleanup":"confirmed","diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a206669"}}],"pending_attempts":[],"fingerprint":"1281ad05e7b6f093a6f0a3d4d775c0f2191fe261941bcbecd951c927fe8ad727"},"owner_checkpoint_fingerprint":"c0d0edf62656295dde8db953170c12734322f9c3e9d69cd4c4a3918a361e7209","structural_binding":null,"pair_fingerprint":"397ad25b75e48cb82a22115bee658b819900fa2c6fad9488a8bc8b549145d1ea"}"##;
const COMPLETED_OWNER_STATE_DOCUMENT: &str = r##"{"trial_rows":[{"owner_ref":"a0","status":"completed"}],"sampler":{"kind":"random","rng_state_hex":"726e673a31"},"pending_params":{},"config":{"sampler":"random","seed":42}}"##;
const COMPLETED_OWNER_STATE: &str =
    "4fe981fcd498a58f957eda98c0abfd04058e8353078b3d645fa834f9397a8feb";
const COMPLETED_PAIR_FINGERPRINT: &str =
    "5b1e6966ce5ae35c94cf6726af8ec574cc6dc531fe2da37f30888d5a7c453a76";
const COMPLETED_CHECKPOINT_FINGERPRINT: &str =
    "55fe15255fdf48be1b5a59abbd62a755d1f26b88e9e534b6d4283c3dcb5be5c5";
/// A sealed completed trial with native float score evidence.
const COMPLETED_PUBLICATION: &str = r##"{"checkpoint":{"schema_version":2,"checkpoint":{"schema_version":1,"binding":{"objective_fingerprint":"1111111111111111111111111111111111111111111111111111111111111111","graph_fingerprint":"2222222222222222222222222222222222222222222222222222222222222222","controller_fingerprint":"3333333333333333333333333333333333333333333333333333333333333333","campaign_fingerprint":"4444444444444444444444444444444444444444444444444444444444444444","fold_set_fingerprint":"5555555555555555555555555555555555555555555555555555555555555555","data_fingerprint":"fixture:data"},"trials":[{"state":"complete","evidence":{"trial_index":0,"params":{"n_components":2},"score":0.123456789,"variant_id":"host_hpo:trial:0000000000","scores":{"schema_version":2,"plan_id":"plan:host_hpo:golden","reports":[{"prediction_id":"pred:model:pls","producer_node":"model:pls","producer_port":"pred","variant_id":"host_hpo:trial:0000000000","partition":"validation","fold_id":"avg","level":"sample","row_count":2,"target_width":1,"target_names":["y"],"metrics":{"mae":1.0,"r2":-0.25,"rmse":0.123456789}}]}}}],"fingerprint":"428a6ee1480a0c6abaab8e153c297fea8a17dbe8e66cd667f11cbf71c8edf5b5"},"source_digest":"a46fbd129a3f4265ddb6515a22f39f72d14de5b71faecbbce7b3f938cfe206e2","attempts":[{"trial_index":0,"capture_id":"capture:first","scope_id":"declaration:search","search_scope_id":"search:capture","owner_ref_hex":"a0","owner_evidence_hex":"6f776e65722d747269616c3a30","phase_index":null,"seed_declarations":[{"seed_id":"seed:sampler","requested":null}],"seed_resolutions":[{"seed_id":"seed:sampler","effective":{"kind":"u64","value":42},"resolution":"generated","parent":null,"policy":{"id":"policy:sampler","version":"1","context_hex":"706f6c6963792d636f6e74657874"},"first_actual_use_id":"choice:n_components"}],"seed_uses":[],"located_choices":[{"choice_id":"choice:n_components","parameter":"n_components","node_id":"model:pls","source_path_hex":["6e5f636f6d706f6e656e7473"],"value_hex":"32","semantics":"stochastic","seed_id":"seed:sampler"}],"candidate":{"owner_variant_id":"host_hpo:trial:0000000000","owner_fingerprint":"9999999999999999999999999999999999999999999999999999999999999999","active_node_ids":["model:pls","transform:snv"],"parameters_hex":"7b226e5f636f6d706f6e656e7473223a327d"},"failure_stage":null,"causal_diagnostic_hex":null,"terminal_capture_id":"capture:first","terminal_scope_id":"declaration:search","owner_observation":{"owner_ref_hex":"a0","ownership":"confirmed","status":"completed","cleanup":"not_attempted","diagnostic_hex":null}}],"pending_attempts":[],"fingerprint":"55fe15255fdf48be1b5a59abbd62a755d1f26b88e9e534b6d4283c3dcb5be5c5"},"owner_checkpoint_fingerprint":"4fe981fcd498a58f957eda98c0abfd04058e8353078b3d645fa834f9397a8feb","structural_binding":null,"pair_fingerprint":"5b1e6966ce5ae35c94cf6726af8ec574cc6dc531fe2da37f30888d5a7c453a76"}"##;
/// Two attempts of one capture that disagree about a restored seed.
const SEED_CONFLICT_PUBLICATION: &str = r##"{"checkpoint":{"schema_version":2,"checkpoint":{"schema_version":1,"binding":{"objective_fingerprint":"1111111111111111111111111111111111111111111111111111111111111111","graph_fingerprint":"2222222222222222222222222222222222222222222222222222222222222222","controller_fingerprint":"3333333333333333333333333333333333333333333333333333333333333333","campaign_fingerprint":"4444444444444444444444444444444444444444444444444444444444444444","fold_set_fingerprint":"5555555555555555555555555555555555555555555555555555555555555555","data_fingerprint":"fixture:data"},"trials":[{"state":"failed","trial_index":0,"params":{"alpha":0.1,"n_components":2,"scale":1.0,"shift":-2.5},"variant_id":"host_hpo:trial:0000000000","error":"runtime validation failed: fit failed"},{"state":"failed","trial_index":1,"params":{"alpha":0.1,"n_components":2,"scale":1.0,"shift":-2.5},"variant_id":"host_hpo:trial:0000000001","error":"runtime validation failed: fit failed"}],"fingerprint":"a497caa49a3026c41b0eff5048142cc89bf58e20d97446ddb0fb78a2607a86b9"},"source_digest":"a46fbd129a3f4265ddb6515a22f39f72d14de5b71faecbbce7b3f938cfe206e2","attempts":[{"trial_index":0,"capture_id":"capture:first","scope_id":"declaration:search","search_scope_id":"search:capture","owner_ref_hex":"a0","owner_evidence_hex":"6f776e65722d747269616c3a30","phase_index":null,"seed_declarations":[{"seed_id":"seed:sampler","requested":null}],"seed_resolutions":[{"seed_id":"seed:sampler","effective":{"kind":"u64","value":42},"resolution":"generated","parent":null,"policy":{"id":"policy:sampler","version":"1","context_hex":"706f6c6963792d636f6e74657874"},"first_actual_use_id":"choice:n_components"}],"seed_uses":[],"located_choices":[{"choice_id":"choice:n_components","parameter":"n_components","node_id":"model:pls","source_path_hex":["6e5f636f6d706f6e656e7473"],"value_hex":"32","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:alpha","parameter":"alpha","node_id":"model:pls","source_path_hex":["616c706861"],"value_hex":"302e31","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:scale","parameter":"scale","node_id":"model:pls","source_path_hex":["7363616c65"],"value_hex":"312e30","semantics":"deterministic","seed_id":null},{"choice_id":"choice:shift","parameter":"shift","node_id":"model:pls","source_path_hex":["7368696674"],"value_hex":"2d322e35","semantics":"deterministic","seed_id":null}],"candidate":{"owner_variant_id":"host_hpo:trial:0000000000","owner_fingerprint":"9999999999999999999999999999999999999999999999999999999999999999","active_node_ids":["model:pls","transform:snv"],"parameters_hex":"7b22616c706861223a302e312c226e5f636f6d706f6e656e7473223a322c227363616c65223a312e302c227368696674223a2d322e357d"},"failure_stage":"evaluation","causal_diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a20666974206661696c6564","terminal_capture_id":"capture:first","terminal_scope_id":"declaration:search","owner_observation":{"owner_ref_hex":"a0","ownership":"confirmed","status":"failed","cleanup":"confirmed","diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a206669"}},{"trial_index":1,"capture_id":"capture:first","scope_id":"declaration:search","search_scope_id":"search:capture","owner_ref_hex":"a1","owner_evidence_hex":"6f776e65722d747269616c3a31","phase_index":null,"seed_declarations":[{"seed_id":"seed:sampler","requested":null}],"seed_resolutions":[{"seed_id":"seed:sampler","effective":{"kind":"u64","value":43},"resolution":"generated","parent":null,"policy":{"id":"policy:sampler","version":"1","context_hex":"706f6c6963792d636f6e74657874"},"first_actual_use_id":"choice:n_components"}],"seed_uses":[],"located_choices":[{"choice_id":"choice:n_components","parameter":"n_components","node_id":"model:pls","source_path_hex":["6e5f636f6d706f6e656e7473"],"value_hex":"32","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:alpha","parameter":"alpha","node_id":"model:pls","source_path_hex":["616c706861"],"value_hex":"302e31","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:scale","parameter":"scale","node_id":"model:pls","source_path_hex":["7363616c65"],"value_hex":"312e30","semantics":"deterministic","seed_id":null},{"choice_id":"choice:shift","parameter":"shift","node_id":"model:pls","source_path_hex":["7368696674"],"value_hex":"2d322e35","semantics":"deterministic","seed_id":null}],"candidate":{"owner_variant_id":"host_hpo:trial:0000000001","owner_fingerprint":"9999999999999999999999999999999999999999999999999999999999999999","active_node_ids":["model:pls","transform:snv"],"parameters_hex":"7b22616c706861223a302e312c226e5f636f6d706f6e656e7473223a322c227363616c65223a312e302c227368696674223a2d322e357d"},"failure_stage":"evaluation","causal_diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a20666974206661696c6564","terminal_capture_id":"capture:first","terminal_scope_id":"declaration:search","owner_observation":{"owner_ref_hex":"a1","ownership":"confirmed","status":"failed","cleanup":"confirmed","diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a20666974206661696c6564"}}],"pending_attempts":[],"fingerprint":"258543e40bfe6d69d066d1a75b65d04d410437bdf67bc032a9c78c164eb30ee6"},"owner_checkpoint_fingerprint":"c2c3104f0370387e43922c27e552b593edf0b989905d15b27c4007d53ddba459","structural_binding":null,"pair_fingerprint":"c3186c65efeb8881eaffcbb001abb319c04ffbdaf7d92217577625337911af45"}"##;

/// Attempts of two lineages that disagree about one logical seed.
const SEED_CONFLICT_LINEAGE_PUBLICATION: &str = r##"{"checkpoint":{"schema_version":2,"checkpoint":{"schema_version":1,"binding":{"objective_fingerprint":"1111111111111111111111111111111111111111111111111111111111111111","graph_fingerprint":"2222222222222222222222222222222222222222222222222222222222222222","controller_fingerprint":"3333333333333333333333333333333333333333333333333333333333333333","campaign_fingerprint":"4444444444444444444444444444444444444444444444444444444444444444","fold_set_fingerprint":"5555555555555555555555555555555555555555555555555555555555555555","data_fingerprint":"fixture:data"},"trials":[{"state":"failed","trial_index":0,"params":{"alpha":0.1,"n_components":2,"scale":1.0,"shift":-2.5},"variant_id":"host_hpo:trial:0000000000","error":"runtime validation failed: fit failed"},{"state":"failed","trial_index":1,"params":{"alpha":0.1,"n_components":2,"scale":1.0,"shift":-2.5},"variant_id":"host_hpo:trial:0000000001","error":"runtime validation failed: fit failed"}],"fingerprint":"a497caa49a3026c41b0eff5048142cc89bf58e20d97446ddb0fb78a2607a86b9"},"source_digest":"a46fbd129a3f4265ddb6515a22f39f72d14de5b71faecbbce7b3f938cfe206e2","attempts":[{"trial_index":0,"capture_id":"capture:first","scope_id":"declaration:search","search_scope_id":"search:capture","owner_ref_hex":"a0","owner_evidence_hex":"6f776e65722d747269616c3a30","phase_index":null,"seed_declarations":[{"seed_id":"seed:sampler","requested":null}],"seed_resolutions":[{"seed_id":"seed:sampler","effective":{"kind":"u64","value":42},"resolution":"generated","parent":null,"policy":{"id":"policy:sampler","version":"1","context_hex":"706f6c6963792d636f6e74657874"},"first_actual_use_id":"choice:n_components"}],"seed_uses":[],"located_choices":[{"choice_id":"choice:n_components","parameter":"n_components","node_id":"model:pls","source_path_hex":["6e5f636f6d706f6e656e7473"],"value_hex":"32","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:alpha","parameter":"alpha","node_id":"model:pls","source_path_hex":["616c706861"],"value_hex":"302e31","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:scale","parameter":"scale","node_id":"model:pls","source_path_hex":["7363616c65"],"value_hex":"312e30","semantics":"deterministic","seed_id":null},{"choice_id":"choice:shift","parameter":"shift","node_id":"model:pls","source_path_hex":["7368696674"],"value_hex":"2d322e35","semantics":"deterministic","seed_id":null}],"candidate":{"owner_variant_id":"host_hpo:trial:0000000000","owner_fingerprint":"9999999999999999999999999999999999999999999999999999999999999999","active_node_ids":["model:pls","transform:snv"],"parameters_hex":"7b22616c706861223a302e312c226e5f636f6d706f6e656e7473223a322c227363616c65223a312e302c227368696674223a2d322e357d"},"failure_stage":"evaluation","causal_diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a20666974206661696c6564","terminal_capture_id":"capture:first","terminal_scope_id":"declaration:search","owner_observation":{"owner_ref_hex":"a0","ownership":"confirmed","status":"failed","cleanup":"confirmed","diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a206669"}},{"trial_index":1,"capture_id":"capture:zero","scope_id":"declaration:search","search_scope_id":"search:capture","owner_ref_hex":"a1","owner_evidence_hex":"6f776e65722d747269616c3a31","phase_index":null,"seed_declarations":[{"seed_id":"seed:sampler","requested":null}],"seed_resolutions":[{"seed_id":"seed:sampler","effective":{"kind":"u64","value":43},"resolution":"generated","parent":null,"policy":{"id":"policy:sampler","version":"1","context_hex":"706f6c6963792d636f6e74657874"},"first_actual_use_id":"choice:n_components"}],"seed_uses":[],"located_choices":[{"choice_id":"choice:n_components","parameter":"n_components","node_id":"model:pls","source_path_hex":["6e5f636f6d706f6e656e7473"],"value_hex":"32","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:alpha","parameter":"alpha","node_id":"model:pls","source_path_hex":["616c706861"],"value_hex":"302e31","semantics":"stochastic","seed_id":"seed:sampler"},{"choice_id":"choice:scale","parameter":"scale","node_id":"model:pls","source_path_hex":["7363616c65"],"value_hex":"312e30","semantics":"deterministic","seed_id":null},{"choice_id":"choice:shift","parameter":"shift","node_id":"model:pls","source_path_hex":["7368696674"],"value_hex":"2d322e35","semantics":"deterministic","seed_id":null}],"candidate":{"owner_variant_id":"host_hpo:trial:0000000001","owner_fingerprint":"9999999999999999999999999999999999999999999999999999999999999999","active_node_ids":["model:pls","transform:snv"],"parameters_hex":"7b22616c706861223a302e312c226e5f636f6d706f6e656e7473223a322c227363616c65223a312e302c227368696674223a2d322e357d"},"failure_stage":"evaluation","causal_diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a20666974206661696c6564","terminal_capture_id":"capture:zero","terminal_scope_id":"declaration:search","owner_observation":{"owner_ref_hex":"a1","ownership":"confirmed","status":"failed","cleanup":"confirmed","diagnostic_hex":"72756e74696d652076616c69646174696f6e206661696c65643a20666974206661696c6564"}}],"pending_attempts":[],"fingerprint":"4dd52b1f158b67eb4c91fce0b7968fce02a537f2f95ecd4440303d8060ded213"},"owner_checkpoint_fingerprint":"c2c3104f0370387e43922c27e552b593edf0b989905d15b27c4007d53ddba459","structural_binding":null,"pair_fingerprint":"e44df7d68c1d5d5e07c73f8986a6d46d1166977cd9fe3cdb41c79a3a0145e0bc"}"##;
/// Pair of `PUBLICATION` with `extended_binding()`, from the SDK encoder.
const EXTENDED_PAIR_FINGERPRINT: &str =
    "6ff141019301eaf8082ca15a21f66af7956d1d47130c2e55dba141e4540f313b";
/// Pair of `PUBLICATION` with a null key, which is not an omitted key.
const NULL_KEY_PAIR_FINGERPRINT: &str =
    "44474672c20a1579f54b8276cfd00463ffb3792b7ea07eb051a5ecd3050ac763";

/// Signed zero, exponent forms, a subnormal, a whole float, an integer and text
/// that is not NFC. Rust float literals keep these exact.
fn extended_binding() -> Value {
    json!({
        "exponent": 1.5e300_f64,
        "integer": 7,
        "negative_zero": -0.0_f64,
        "subnormal": 5e-324_f64,
        "text": "e\u{301}",
        "whole": 1.0_f64
    })
}

fn publication() -> HostHpoCapturePublication {
    serde_json::from_str(PUBLICATION).unwrap()
}

fn request(capture_id: &str, failed_attempts: Vec<FailedAttempt>) -> HostHpoCaptureSearchRequest {
    serde_json::from_value(json!({
        "schema_version": 2,
        "request": {
            "target_node": "model:pls", "trial_budget": 3, "metric": "rmse",
            "direction": "minimize", "optimizer_descriptor": {"sampler": "random", "seed": 42}
        },
        "capture": {
            "capture_id": capture_id, "scope_id": "declaration:search",
            "search_scope_id": "search:capture", "declaration_text": DECLARATION,
            "source_digest": DECLARATION_DIGEST, "node_ids": ["transform:snv", "model:pls"],
            "seed_declarations": [{"seed_id": "seed:sampler", "requested": null}],
            "failed_attempts": failed_attempts
        }
    }))
    .unwrap()
}

fn hex(text: &str) -> String {
    text.bytes().map(|byte| format!("{byte:02x}")).collect()
}

/// A proposal that the closed recorder accepts.
fn proposal() -> Value {
    json!({"schema_version": 1, "outcome": {
        "status": "proposed", "owner_ref_hex": "a0", "owner_evidence_hex": hex("owner-trial:0"),
        "params": {"n_components": 2},
        "located_choices": [{
            "choice_id": "choice:n_components", "parameter": "n_components",
            "node_id": "model:pls", "source_path_hex": [hex("n_components")],
            "value_hex": hex("2"), "semantics": "stochastic", "seed_id": "seed:sampler"
        }],
        "seed_resolutions": [{
            "seed_id": "seed:sampler", "effective": {"kind": "u64", "value": 42},
            "resolution": "generated", "parent": null,
            "policy": {"id": "policy:sampler", "version": "1", "context_hex": hex("context")},
            "first_actual_use_id": "choice:n_components"
        }],
        "seed_uses": []
    }})
}

fn proposal_verdict(document: Value) -> std::result::Result<(), String> {
    let proposal: HostHpoProposalEvidence = serde_json::from_value(document).unwrap();
    validate_host_hpo_capture_proposal(&request("capture:first", Vec::new()), &proposal)
        .map_err(|error| error.to_string())
}

#[test]
fn pair_golden_matches_the_tcv1_encoder_including_binary64_values() {
    let publication = publication();
    publication.checkpoint.verify_seal().unwrap();
    publication.checkpoint.checkpoint.verify_seal().unwrap();
    assert_eq!(publication.checkpoint.fingerprint, CHECKPOINT_FINGERPRINT);
    assert_eq!(
        host_hpo_capture_pair_fingerprint(&publication.checkpoint, OWNER_STATE, None).unwrap(),
        PAIR_FINGERPRINT
    );
    assert_eq!(
        publication.pair_fingerprint.as_deref(),
        Some(PAIR_FINGERPRINT)
    );
}

#[test]
fn completed_float_score_pair_golden_matches_the_sdk_encoder() {
    let publication: HostHpoCapturePublication =
        serde_json::from_str(COMPLETED_PUBLICATION).unwrap();
    publication.checkpoint.verify_seal().unwrap();
    publication.checkpoint.checkpoint.verify_seal().unwrap();
    assert_eq!(
        publication.checkpoint.fingerprint,
        COMPLETED_CHECKPOINT_FINGERPRINT
    );
    let HostHpoTerminalTrial::Complete { evidence } = &publication.checkpoint.checkpoint.trials[0]
    else {
        panic!("the golden holds a completed trial");
    };
    evidence.scores.validate().unwrap();
    assert_eq!(evidence.score, evidence.scores.reports[0].metrics["rmse"]);
    assert_eq!(evidence.scores.reports[0].metrics["mae"], 1.0);
    assert_eq!(
        host_hpo_capture_pair_fingerprint(&publication.checkpoint, COMPLETED_OWNER_STATE, None)
            .unwrap(),
        COMPLETED_PAIR_FINGERPRINT
    );
    assert_eq!(
        publication.pair_fingerprint.as_deref(),
        Some(COMPLETED_PAIR_FINGERPRINT)
    );
    let resumed = request("capture:second", Vec::new());
    let observed = HostHpoProposalState {
        owner_checkpoint_fingerprint: Some(COMPLETED_OWNER_STATE.into()),
        structural_binding: None,
        owner_trial_count: 1,
    };
    validate_host_hpo_capture_lineage(&resumed, Some(&publication), Some(&observed)).unwrap();
}

#[test]
fn owner_state_digest_golden_covers_rows_sampler_pending_and_config() {
    for (document, digest) in [
        (OWNER_STATE_DOCUMENT, OWNER_STATE),
        (COMPLETED_OWNER_STATE_DOCUMENT, COMPLETED_OWNER_STATE),
    ] {
        assert_eq!(
            tcv1_sha256(&parse_typed_json(document).unwrap()).unwrap(),
            digest
        );
        let state: Value = serde_json::from_str(document).unwrap();
        for key in ["trial_rows", "sampler", "pending_params", "config"] {
            assert!(state.get(key).is_some(), "the digest must cover {key}");
        }
    }
    let mut moved: Value = serde_json::from_str(OWNER_STATE_DOCUMENT).unwrap();
    moved["sampler"]["rng_state_hex"] = json!(hex("rng:2"));
    assert_ne!(
        tcv1_sha256(&parse_typed_json(&moved.to_string()).unwrap()).unwrap(),
        OWNER_STATE,
        "equal rows with a moved sampler are different owner state"
    );
}

#[test]
fn restored_attempts_that_disagree_about_a_seed_refuse() {
    let publication: HostHpoCapturePublication =
        serde_json::from_str(SEED_CONFLICT_PUBLICATION).unwrap();
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();
    let error = validate_host_hpo_capture_lineage(
        &request("capture:second", derived),
        Some(&publication),
        None,
    )
    .unwrap_err();
    assert_eq!(error.code(), "host_hpo_predecessor_refused");
    assert!(error.to_string().contains("seed_conflict"), "{error}");
}

#[test]
fn closed_contracts_refuse_unknown_and_missing_fields() {
    let exhausted = json!({"schema_version": 1, "outcome": {"status": "exhausted"}});
    assert!(serde_json::from_value::<HostHpoProposalEvidence>(exhausted).is_ok());
    let extra = json!({"schema_version": 1, "outcome": {"status": "exhausted", "extra": 1}});
    assert!(
        serde_json::from_value::<HostHpoProposalEvidence>(extra).is_err(),
        "exhausted is closed"
    );
    let document: Value = serde_json::from_str(PUBLICATION).unwrap();
    assert!(serde_json::from_value::<HostHpoCapturePublication>(document.clone()).is_ok());
    let mut unknown = document.clone();
    unknown["checkpoint"]["attempts"][0]["extra"] = json!(1);
    assert!(serde_json::from_value::<HostHpoCapturePublication>(unknown).is_err());
    for field in [
        "search_scope_id",
        "terminal_capture_id",
        "terminal_scope_id",
    ] {
        let mut missing = document.clone();
        missing["checkpoint"]["attempts"][0]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(
            serde_json::from_value::<HostHpoCapturePublication>(missing).is_err(),
            "{field} is required"
        );
    }
}

#[test]
fn extended_canonical_golden_matches_the_sdk_encoder() {
    let checkpoint = publication().checkpoint;
    let binding = extended_binding();
    assert_eq!(
        host_hpo_capture_pair_fingerprint(&checkpoint, OWNER_STATE, Some(&binding)).unwrap(),
        EXTENDED_PAIR_FINGERPRINT
    );
    let mut normalized = binding;
    normalized["text"] = json!("\u{e9}");
    normalized["negative_zero"] = json!(0.0_f64);
    assert_eq!(
        host_hpo_capture_pair_fingerprint(&checkpoint, OWNER_STATE, Some(&normalized)).unwrap(),
        EXTENDED_PAIR_FINGERPRINT,
        "NFC text and signed zero normalize"
    );
}

#[test]
fn absent_structural_binding_is_omitted_from_the_pair_preimage() {
    let checkpoint = publication().checkpoint;
    assert_eq!(
        host_hpo_capture_pair_fingerprint(&checkpoint, OWNER_STATE, None).unwrap(),
        PAIR_FINGERPRINT
    );
    assert_eq!(
        host_hpo_capture_pair_fingerprint(&checkpoint, OWNER_STATE, Some(&Value::Null)).unwrap(),
        NULL_KEY_PAIR_FINGERPRINT,
        "a present null is a different preimage"
    );
    assert_ne!(NULL_KEY_PAIR_FINGERPRINT, PAIR_FINGERPRINT);
}

#[test]
fn seed_value_is_immutable_across_lineages() {
    let publication: HostHpoCapturePublication =
        serde_json::from_str(SEED_CONFLICT_LINEAGE_PUBLICATION).unwrap();
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();
    let error = validate_host_hpo_capture_lineage(
        &request("capture:second", derived),
        Some(&publication),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("seed_conflict"), "{error}");
}

#[test]
fn a_new_declaration_cannot_change_a_restored_seed_request() {
    let publication = publication();
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();
    let mut resumed = request("capture:second", derived);
    resumed.capture.seed_declarations[0].requested =
        Some(serde_json::from_value(json!({"kind": "u64", "value": 7})).unwrap());
    let error = validate_host_hpo_capture_lineage(&resumed, Some(&publication), None).unwrap_err();
    assert!(error.to_string().contains("seed_conflict"), "{error}");
}

#[test]
fn a_proposal_cannot_state_one_seed_twice() {
    let mut document = proposal();
    let second = document["outcome"]["seed_resolutions"][0].clone();
    document["outcome"]["seed_resolutions"]
        .as_array_mut()
        .unwrap()
        .push(second);
    document["outcome"]["seed_resolutions"][1]["effective"]["value"] = json!(43);
    let detail = proposal_verdict(document).unwrap_err();
    assert!(detail.contains("seed_resolution_conflict"), "{detail}");
}

#[test]
fn pair_keeps_integer_and_binary64_tokens_distinct() {
    let mut changed = publication().checkpoint;
    let HostHpoTerminalTrial::Failed { params, .. } = &mut changed.checkpoint.trials[0] else {
        panic!("the golden holds a failed trial");
    };
    assert_eq!(params["scale"], json!(1.0));
    params.insert("scale".into(), json!(1));
    assert_ne!(
        host_hpo_capture_pair_fingerprint(&changed, OWNER_STATE, None).unwrap(),
        PAIR_FINGERPRINT
    );
}

#[test]
fn pair_covers_owner_state_and_structural_binding() {
    let checkpoint = publication().checkpoint;
    assert_ne!(
        host_hpo_capture_pair_fingerprint(&checkpoint, "owner-state:moved", None).unwrap(),
        PAIR_FINGERPRINT
    );
    assert_ne!(
        host_hpo_capture_pair_fingerprint(&checkpoint, OWNER_STATE, Some(&json!({"a": 1})))
            .unwrap(),
        PAIR_FINGERPRINT
    );
}

#[test]
fn derived_provenance_pairs_exactly_and_passes_the_closed_gate() {
    let publication = publication();
    let derived = host_hpo_capture_prefix_provenance(&publication).unwrap();
    assert_eq!(derived.len(), 1);
    let prefix = derived[0].trusted_prefix_ref.as_ref().unwrap();
    assert_eq!(prefix.pair_fingerprint.as_deref(), Some(PAIR_FINGERPRINT));
    let resumed = request("capture:second", derived);
    resumed.validate().unwrap();
    let observed = HostHpoProposalState {
        owner_checkpoint_fingerprint: Some(OWNER_STATE.into()),
        structural_binding: None,
        owner_trial_count: 1,
    };
    validate_host_hpo_capture_lineage(&resumed, Some(&publication), Some(&observed)).unwrap();
}

#[test]
fn null_pair_has_no_provenance_and_refuses_resume() {
    let mut publication = publication();
    publication.pair_fingerprint = None;
    assert!(host_hpo_capture_prefix_provenance(&publication).is_err());
    let error = validate_host_hpo_capture_lineage(
        &request("capture:second", Vec::new()),
        Some(&publication),
        None,
    )
    .unwrap_err();
    assert_eq!(error.code(), "host_hpo_predecessor_refused");
    assert!(error.to_string().contains("pair_unavailable"), "{error}");
}

#[test]
fn closed_recorder_judges_literal_proposals() {
    proposal_verdict(proposal()).unwrap();
    let mut a1 = proposal();
    a1["outcome"]["seed_resolutions"][0]["policy"] = json!(null);
    let mut a2 = proposal();
    a2["outcome"]["seed_resolutions"][0]["first_actual_use_id"] = json!("use:sampler");
    let mut a3 = proposal();
    a3["outcome"]["located_choices"][0]["semantics"] = json!("deterministic");
    let mut a4 = proposal();
    a4["outcome"]["located_choices"][0]["semantics"] = json!("deterministic");
    a4["outcome"]["located_choices"][0]["seed_id"] = json!(null);
    let mut undeclared = proposal();
    undeclared["outcome"]["located_choices"][0]["node_id"] = json!("model:undeclared");
    for (document, code) in [
        (a1, "seed_resolution"),
        (a2, "seed_use_binding"),
        (a3, "choice_seed"),
        (a4, "unused_effective_seed"),
        (undeclared, "unresolved_reference"),
    ] {
        let detail = proposal_verdict(document).unwrap_err();
        assert!(detail.contains(code), "expected {code}, observed {detail}");
    }
}

fn opened_recorder() -> Recorder<std::io::Sink> {
    let request = request("capture:gate", Vec::new());
    let mut recorder = Recorder::new(std::io::sink(), "capture:gate", Limits::default()).unwrap();
    for event in [
        json!({"kind": "declaration", "scope_id": "declaration:search",
            "scope_kind": "declaration", "parent_scope_id": null,
            "declaration_text": DECLARATION, "source_digest": DECLARATION_DIGEST,
            "node_ids": request.capture.node_ids}),
        json!({"kind": "capture_provenance", "scope_id": "declaration:search",
            "failed_attempts": []}),
        json!({"kind": "scope_open", "scope_id": "search:capture", "scope_kind": "search",
            "parent_scope_id": "declaration:search",
            "declaration_ref": {"scope_id": "declaration:search",
                "source_digest": DECLARATION_DIGEST}}),
    ] {
        recorder.record(&event_from(event)).unwrap();
    }
    recorder
}

fn event_from(value: Value) -> Event {
    serde_json::from_value(value).unwrap()
}

#[test]
fn closed_recorder_refuses_operation_error_closure_without_an_operation_failure() {
    let mut recorder = opened_recorder();
    let refused = recorder
        .record(&event_from(
            json!({"kind": "search_close", "scope_id": "search:capture",
            "reason": "failed", "unvisited": null, "basis": "operation_error",
            "evidence": {"availability": "known", "value_hex": hex("cause")}}),
        ))
        .unwrap_err();
    assert_eq!(refused.code(), "closure_reason");
}

#[test]
fn abort_cannot_finish_with_an_open_search() {
    let mut recorder = opened_recorder();
    assert_eq!(recorder.finish().unwrap_err().code(), "empty_or_unclosed");
}

#[cfg(dag_ml_workspace_contract_fixtures)]
mod literal_vectors {
    use super::*;

    fn assert_vector(vector: &Value, observed: std::result::Result<(), String>) {
        match (vector["expect"].as_str().unwrap(), observed) {
            ("accepted", Ok(())) | ("schema", Err(_)) => {}
            (code, Err(detail)) if detail.contains(code) => {}
            (code, observed) => panic!("{}: expected {code}, observed {observed:?}", vector["id"]),
        }
    }

    fn parsed<T: serde::de::DeserializeOwned>(value: &Value) -> std::result::Result<T, String> {
        serde_json::from_value(value.clone()).map_err(|error| error.to_string())
    }

    fn text<T>(result: Result<T>) -> std::result::Result<T, String> {
        result.map_err(|error| error.to_string())
    }

    #[test]
    fn literal_capture_vectors_match_the_owner_types() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../examples/fixtures/training/host_hpo_capture.v1.json"
        ))
        .unwrap();
        assert_eq!(
            fixture["planning_sibling_diagnostic"],
            HOST_HPO_PLANNING_SIBLING_DIAGNOSTIC
        );
        assert_eq!(
            fixture["pair_goldens"][0]["pair_fingerprint"],
            PAIR_FINGERPRINT
        );
        assert_eq!(
            fixture["pair_goldens"][1]["pair_fingerprint"],
            COMPLETED_PAIR_FINGERPRINT
        );
        assert_eq!(
            fixture["pair_goldens"][2]["pair_fingerprint"],
            EXTENDED_PAIR_FINGERPRINT
        );
        let base: HostHpoCaptureSearchRequest =
            parsed(&fixture["resume_fixture"]["request"]).unwrap();
        for vector in fixture["proposal_vectors"].as_array().unwrap() {
            let observed = parsed::<HostHpoProposalEvidence>(&vector["document"])
                .and_then(|proposal| text(validate_host_hpo_capture_proposal(&base, &proposal)));
            assert_vector(vector, observed);
        }
        for vector in fixture["attempt_vectors"].as_array().unwrap() {
            assert_vector(
                vector,
                parsed::<HostHpoAttemptEvidence>(&vector["document"]).map(|_| ()),
            );
        }
        for vector in fixture["proposal_state_vectors"].as_array().unwrap() {
            let observed = parsed::<HostHpoProposalState>(&vector["document"]).and_then(|state| {
                let fresh = HostHpoProposalState {
                    owner_trial_count: 0,
                    ..state
                };
                text(validate_host_hpo_capture_lineage(&base, None, Some(&fresh)))
            });
            assert_vector(vector, observed);
        }
        for vector in fixture["request_vectors"].as_array().unwrap() {
            let observed = parsed::<HostHpoCaptureSearchRequest>(&vector["document"])
                .and_then(|request| text(validate_host_hpo_capture_lineage(&request, None, None)));
            assert_vector(vector, observed);
        }
        for vector in fixture["failed_attempt_vectors"].as_array().unwrap() {
            let observed = parsed::<FailedAttempt>(&vector["document"]).and_then(|failed| {
                let mut request = base.clone();
                request.capture.failed_attempts = vec![failed];
                text(request.validate())
            });
            assert_vector(vector, observed);
        }
        for vector in fixture["publication_vectors"].as_array().unwrap() {
            let observed =
                parsed::<HostHpoCapturePublication>(&vector["document"]).and_then(|publication| {
                    let mut request = base.clone();
                    request.capture.failed_attempts =
                        host_hpo_capture_prefix_provenance(&publication).unwrap_or_default();
                    text(validate_host_hpo_capture_lineage(
                        &request,
                        Some(&publication),
                        None,
                    ))
                });
            assert_vector(vector, observed);
        }
        let restored: HostHpoCapturePublication =
            parsed(&fixture["resume_fixture"]["publication"]).unwrap();
        let pending: HostHpoCapturePublication =
            parsed(&fixture["resume_fixture"]["pending_publication"]).unwrap();
        for vector in fixture["resume_vectors"].as_array().unwrap() {
            let mut request = base.clone();
            request.capture.capture_id = vector["capture_id"].as_str().unwrap().to_owned();
            request.capture.failed_attempts = parsed(&vector["failed_attempts"]).unwrap();
            if let Some(trial_budget) = vector["trial_budget"].as_u64() {
                request.request.trial_budget = trial_budget as u32;
            }
            if vector.get("seed_declarations").is_some() {
                request.capture.seed_declarations = parsed(&vector["seed_declarations"]).unwrap();
            }
            let chosen = if vector["publication"] == "pending" {
                &pending
            } else {
                &restored
            };
            let publication = vector["restored"].as_bool().unwrap().then_some(chosen);
            let observed: Option<HostHpoProposalState> = parsed(&vector["observed"]).unwrap();
            let verdict = text(validate_host_hpo_capture_lineage(
                &request,
                publication,
                observed.as_ref(),
            ));
            assert_vector(vector, verdict);
        }
    }
}
