//! Exact per-fixture mutation selectors; no preceding negative can mask the target.
use super::assert_negative_fixture;

#[test]
fn mutant_frame_digest() {
    assert_negative_fixture("integrity_frame_digest");
}

#[test]
fn mutant_frame_chain() {
    assert_negative_fixture("integrity_frame_sequence");
}

#[test]
fn mutant_frame_capture_id() {
    assert_negative_fixture("integrity_frame_capture_id");
}

#[test]
fn mutant_byte_length() {
    assert_negative_fixture("integrity_frame_byte_length");
}

#[test]
fn mutant_completion_capture_id() {
    assert_negative_fixture("integrity_completion_capture_id");
}

#[test]
fn mutant_completion_chain() {
    assert_negative_fixture("integrity_completion_frame_count");
}

#[test]
fn mutant_completion_counts() {
    assert_negative_fixture("integrity_completion_counts");
}

#[test]
fn mutant_completion_scopes_digest() {
    assert_negative_fixture("integrity_completion_closed_scopes_digest");
}

#[test]
fn mutant_completion_digest() {
    assert_negative_fixture("integrity_completion_digest");
}

#[test]
fn mutant_pending_trial() {
    assert_negative_fixture("integrity_pending_trial");
}

#[test]
fn mutant_open_children() {
    assert_negative_fixture("integrity_open_children");
}

#[test]
fn mutant_pending_visit_a_zero() {
    assert_negative_fixture("integrity_pending_visit_a_zero");
}

#[test]
fn mutant_trial_status_visit() {
    assert_negative_fixture("integrity_trial_status_visit");
}

#[test]
fn mutant_trial_status_resume() {
    assert_negative_fixture("integrity_trial_status_resume");
}

#[test]
fn mutant_trial_status_outcome_mode() {
    assert_negative_fixture("integrity_trial_status_outcome_mode");
}

#[test]
fn mutant_trial_status_outcome_initial() {
    assert_negative_fixture("integrity_trial_status_outcome_initial");
}

#[test]
fn mutant_visit_disposition() {
    assert_negative_fixture("integrity_visit_disposition");
}

#[test]
fn mutant_effective_seed_requested() {
    assert_negative_fixture("integrity_effective_seed_requested_null");
}

#[test]
fn mutant_unvisited_evidence() {
    assert_negative_fixture("integrity_unvisited_evidence");
}

#[test]
fn mutant_cleanup_observation() {
    assert_negative_fixture("integrity_cleanup_observation");
}

#[test]
fn mutant_provenance_carrier_event() {
    assert_negative_fixture("integrity_provenance_carrier_event");
}

#[test]
fn mutant_provenance_carrier_finish() {
    assert_negative_fixture("integrity_provenance_carrier_finish");
}

#[test]
fn mutant_completed_failure() {
    assert_negative_fixture("integrity_completed_failure_transition");
}

#[test]
fn mutant_enumeration_reason() {
    assert_negative_fixture("integrity_enumeration_count_limit");
}

#[test]
fn mutant_choice_use_namespace() {
    assert_negative_fixture("integrity_choice_use_choice_first");
}

#[test]
fn mutant_sibling_owner_namespace() {
    assert_negative_fixture("integrity_sibling_owner_ref");
}

#[test]
fn mutant_owner_encoded_equality() {
    assert_negative_fixture("integrity_owner_encoded_mismatch_profile");
}

#[test]
fn mutant_outer_previous_grammar() {
    assert_negative_fixture("integrity_outer_frame_previous_digest");
}

#[test]
fn mutant_event_grammar_mapping() {
    assert_negative_fixture("integrity_event_choice_id");
}
