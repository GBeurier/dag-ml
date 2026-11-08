//! Contextual owner regressions with genuine saved synthetic-host roots.
#[path = "support/multi_output_mock.rs"]
mod host;
use dag_ml_core::*;
use std::collections::BTreeMap;
use std::sync::{atomic::Ordering, Arc};

fn roots() -> (TrainingOutcome, PortablePredictorPackage) {
    (
        TrainingOutcome::from_json(include_str!(
            "fixtures/multi_output_target_claims/explain/outcome.json"
        ))
        .unwrap(),
        PortablePredictorPackage::from_json(include_str!(
            "fixtures/multi_output_target_claims/explain/package.json"
        ))
        .unwrap(),
    )
}
fn envelopes(selected: bool, v2: bool) -> BTreeMap<String, ExternalDataPlanEnvelope> {
    let mut values: BTreeMap<String, ExternalDataPlanEnvelope> = serde_json::from_str(
        include_str!("fixtures/multi_output_target_claims/shared/envelopes.json"),
    )
    .unwrap();
    for e in values.values_mut() {
        let old = e.predict_cohort.take().unwrap();
        if v2 {
            e.predict_cohort = Some(
                PredictCohort::from_relations(
                    PredictCohortRole::Inference,
                    old.relations,
                    if selected {
                        vec!["protein".into()]
                    } else {
                        vec!["protein".into(), "fat".into()]
                    },
                    old.data_content_fingerprint,
                    None,
                )
                .unwrap(),
            );
        } else {
            e.schema_version = 1;
        }
    }
    values
}
fn request(source: &TrainingOutcome, phase: Phase, selected: bool) -> TrainingReplayRequest {
    let mut r: TrainingReplayRequest = serde_json::from_str(include_str!(
        "fixtures/multi_output_target_claims/shared/request.json"
    ))
    .unwrap();
    r.source_outcome_fingerprint = source.outcome_fingerprint.clone();
    r.phase = phase;
    if selected {
        r.output_binding_ids = vec!["output:a".into()];
    }
    r.request_fingerprint = r.compute_fingerprint().unwrap();
    r
}
struct Controller {
    id: ControllerId,
    base: Arc<RuntimeControllerRegistry>,
    mode: &'static str,
}
impl RuntimeController for Controller {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }
    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        let mut r = self.base.get(&self.id).unwrap().invoke(task)?;
        for b in &mut r.predictions {
            match self.mode {
                "alien" => b.sample_ids[0] = SampleId::new("sample:alien").unwrap(),
                "old" => b.sample_ids[0] = SampleId::new("sample:1").unwrap(),
                "partial" => {
                    b.sample_ids.truncate(1);
                    b.values.truncate(1);
                }
                "reordered" => {
                    b.sample_ids.reverse();
                    b.values.reverse();
                }
                "duplicate" => {
                    b.sample_ids[1] = b.sample_ids[0].clone();
                }
                _ => {}
            }
        }
        for b in &mut r.explanations {
            match self.mode {
                "target" => b.target_name = Some("wrong".into()),
                "alias" => b.target_name = Some("protein:a".into()),
                "global" => b.target_name = None,
                "raw" => b.payload = serde_json::json!({"nested":[{"FeAtUrE_MaTrIx":[[0.0]]}]}),
                "handle" => b.payload = serde_json::json!({"nested":{"handle":1}}),
                "unselected_raw" if task.node_plan.node_id.as_str() == "model:second" => {
                    b.payload = serde_json::json!({"raw_spectra":[0]})
                }
                "matrix" => b.payload = serde_json::json!({"X":[[1,2],[3,4]]}),
                _ => {}
            }
        }
        if self.mode == "empty" && task.node_plan.node_id.as_str() == "model:base" {
            r.explanations.clear();
        }
        Ok(r)
    }
    fn hydrate_artifact_payload(
        &self,
        r: &ArtifactMaterializationRequest,
        p: &[u8],
    ) -> Result<HandleRef> {
        self.base
            .get(&self.id)
            .unwrap()
            .hydrate_artifact_payload(r, p)
    }
    fn release_hydrated_artifact_payload(&self, h: &HandleRef) -> Result<()> {
        self.base
            .get(&self.id)
            .unwrap()
            .release_hydrated_artifact_payload(h)
    }
}
fn execute(
    phase: Phase,
    selected: bool,
    v2: bool,
    mode: &'static str,
    attached: bool,
) -> (Result<TrainingReplayOutcome>, Arc<host::Counts>) {
    let (source, pkg) = roots();
    let env = envelopes(selected, v2);
    let req = request(&source, phase, selected);
    let counts = Arc::new(host::Counts::default());
    counts.zero();
    let base = Arc::new(host::registry(&pkg, &env, &counts, false));
    let mut registry = RuntimeControllerRegistry::new();
    for id in pkg.effective_plan.controller_manifests.keys() {
        registry
            .register(Box::new(Controller {
                id: id.clone(),
                base: base.clone(),
                mode,
            }))
            .unwrap();
    }
    let provider = host::Provider {
        envelopes: env.clone(),
        counts: counts.clone(),
    };
    let loaded = LoadedPredictor::<HandleRef>::new(pkg, BTreeMap::new()).unwrap();
    let result = if attached {
        execute_attached_training_replay(AttachedTrainingReplayInput {
            source: &source,
            request: &req,
            outcome_id: "outcome:contextual.attached".into(),
            run_id: RunId::new("run:contextual.attached").unwrap(),
            controllers: &registry,
            data_provider: &provider,
            artifact_store: &InMemoryArtifactStore::new(),
            data_envelopes: &env,
            warnings: vec![],
            diagnostics: BTreeMap::new(),
        })
    } else {
        execute_loaded_predictor_replay(LoadedPredictorReplayInput {
            predictor: &loaded,
            request: &req,
            outcome_id: "outcome:contextual.loaded".into(),
            run_id: RunId::new("run:contextual.loaded").unwrap(),
            controllers: &registry,
            data_provider: &provider,
            data_envelopes: &env,
            warnings: vec![],
            diagnostics: BTreeMap::new(),
        })
    };
    (result, counts)
}
#[test]
fn physical_membership_preserves_subset_reordering_and_refuses_alien_old_duplicate() {
    for v2 in [false, true] {
        for attached in [false, true] {
            for mode in ["normal", "partial", "reordered"] {
                execute(Phase::Predict, false, v2, mode, attached)
                    .0
                    .unwrap();
            }
            for mode in ["alien", "old", "duplicate"] {
                let (r, c) = execute(Phase::Predict, false, v2, mode, attached);
                assert!(r.is_err(), "{mode}");
                assert!(c.controller.load(Ordering::SeqCst) > 0);
                assert_eq!(
                    c.hydrate.load(Ordering::SeqCst),
                    c.release.load(Ordering::SeqCst)
                );
            }
        }
    }
}
#[test]
fn stored_intrinsic_contextual_and_current_row_levels_differ_truthfully() {
    let (source, pkg) = roots();
    let req = request(&source, Phase::Predict, false);
    let env = envelopes(false, true);
    let mut result = execute(Phase::Predict, false, true, "normal", false)
        .0
        .unwrap();
    result
        .validate_against_current_cohort(&source, &req, &env)
        .unwrap();
    result
        .validate_against_package_current_cohort(&pkg, &req, &env)
        .unwrap();
    let mut missing = env.clone();
    missing.clear();
    assert!(matches!(
        result.validate_against_current_cohort(&source, &req, &missing),
        Err(ReplayCurrentCohortError::Missing(
            ReplayCurrentCohortMissing::EnvelopeAbsent { .. }
        ))
    ));
    let mut different = env.clone();
    different
        .values_mut()
        .next()
        .unwrap()
        .predict_cohort
        .as_mut()
        .unwrap()
        .data_content_fingerprint = "f".repeat(64);
    let cohort = different
        .values_mut()
        .next()
        .unwrap()
        .predict_cohort
        .as_mut()
        .unwrap();
    cohort.cohort_fingerprint = cohort.fingerprint().unwrap();
    assert!(matches!(
        result.validate_against_current_cohort(&source, &req, &different),
        Err(ReplayCurrentCohortError::Mismatch { .. })
    ));
    let mut forged = env.clone();
    let cohort = forged
        .values_mut()
        .next()
        .unwrap()
        .predict_cohort
        .as_mut()
        .unwrap();
    // Deliberately malformed supplied preimage: no forged/re-signed cohort.
    cohort.relations.records[0].sample_id = SampleId::new("sample:alien").unwrap();
    assert!(matches!(
        result.validate_against_package_current_cohort(&pkg, &req, &forged),
        Err(ReplayCurrentCohortError::Mismatch { .. })
    ));
    // A unit-edited stored replay, never claimed to be a genuine emitted fixture.
    result.outputs[0].predictions[0].sample_ids[0] = SampleId::new("sample:alien").unwrap();
    result.outcome_fingerprint = result.compute_fingerprint().unwrap();
    result.validate().unwrap();
    result.validate_against_package(&pkg, &req).unwrap();
    assert!(matches!(
        result.validate_against_package_current_cohort(&pkg, &req, &env),
        Err(ReplayCurrentCohortError::Mismatch { .. })
    ));
    result.outputs[0].predictions[0].sample_ids[1] = SampleId::new("sample:alien").unwrap();
    result.outcome_fingerprint = result.compute_fingerprint().unwrap();
    assert!(result.validate().is_err());
}
#[test]
fn selected_explanations_keep_lineage_and_validate_all_payloads_before_filtering() {
    for attached in [true, false] {
        for mode in ["normal", "global", "matrix"] {
            let result = execute(Phase::Explain, true, true, mode, attached)
                .0
                .unwrap();
            assert_eq!(result.lineage.len(), 3);
            assert_eq!(result.explanations.len(), 1);
            assert_eq!(result.explanations[0].producer_node.as_str(), "model:base");
        }
        for mode in [
            "target",
            "alias",
            "raw",
            "handle",
            "unselected_raw",
            "empty",
        ] {
            assert!(
                execute(Phase::Explain, true, true, mode, attached)
                    .0
                    .is_err(),
                "{mode}"
            );
        }
    }
    let (source, pkg) = roots();
    let mut result = execute(Phase::Explain, false, true, "normal", false)
        .0
        .unwrap();
    let req = request(&source, Phase::Explain, true);
    result.replay_request_fingerprint = req.request_fingerprint.clone();
    result.outcome_fingerprint = result.compute_fingerprint().unwrap();
    result.validate().unwrap();
    assert!(result.validate_against_package(&pkg, &req).is_err());
}
#[test]
fn v1_relation_digest_tampering_refuses_before_execution() {
    let (source, pkg) = roots();
    let req = request(&source, Phase::Predict, false);
    let mut env = envelopes(false, false);
    env.values_mut()
        .next()
        .unwrap()
        .coordinator_relations
        .as_mut()
        .unwrap()
        .records[0]
        .sample_id = SampleId::new("sample:alien").unwrap();
    assert!(preflight_loaded_predictor_replay(&pkg, &req, &env)
        .unwrap_err()
        .to_string()
        .contains("relation fingerprint"));
}

#[test]
fn newly_generated_v2_fixture_passes_each_additive_current_row_api() {
    let source = TrainingOutcome::from_json(include_str!(
        "fixtures/contextual_replay_v2/training-outcome.json"
    ))
    .unwrap();
    let package = PortablePredictorPackage::from_json(include_str!(
        "fixtures/contextual_replay_v2/package.json"
    ))
    .unwrap();
    for (request, outcome, envelopes) in [
        (
            include_str!("fixtures/contextual_replay_v2/predict-request.json"),
            include_str!("fixtures/contextual_replay_v2/predict-replay.json"),
            include_str!("fixtures/contextual_replay_v2/predict-envelopes.json"),
        ),
        (
            include_str!("fixtures/contextual_replay_v2/explain-selected-request.json"),
            include_str!("fixtures/contextual_replay_v2/explain-selected-replay.json"),
            include_str!("fixtures/contextual_replay_v2/explain-selected-envelopes.json"),
        ),
    ] {
        let r = TrainingReplayRequest::from_json(request).unwrap();
        let o = TrainingReplayOutcome::from_json(outcome).unwrap();
        let e = serde_json::from_str(envelopes).unwrap();
        o.validate_against_current_cohort(&source, &r, &e).unwrap();
        o.validate_against_package_current_cohort(&package, &r, &e)
            .unwrap();
    }
}

#[test]
fn stored_current_cohort_rejects_incomplete_and_unknown_request_keys_without_panicking() {
    let package = PortablePredictorPackage::from_json(include_str!(
        "../../../examples/fixtures/training/portable_predictor_package.v1.json"
    ))
    .unwrap();
    let source = TrainingOutcome::from_json(include_str!(
        "../../../examples/fixtures/training/training_outcome_refit.v1.json"
    ))
    .unwrap();
    let outcome = TrainingReplayOutcome::from_json(include_str!(
        "../../../examples/fixtures/training/replay/training_replay_outcome_predict.v1.json"
    ))
    .unwrap();
    let request = TrainingReplayRequest::from_json(include_str!(
        "../../../examples/fixtures/training/replay/training_replay_request_predict.v1.json"
    ))
    .unwrap();
    let value: serde_json::Value = serde_json::from_str(include_str!(
        "../../../examples/fixtures/training/replay/training_replay_input_envelopes.v1.json"
    ))
    .unwrap();
    let envelopes: BTreeMap<String, ExternalDataPlanEnvelope> =
        serde_json::from_value(value["envelopes"].clone()).unwrap();
    outcome.validate_against(&source, &request).unwrap();
    outcome
        .validate_against_package(&package, &request)
        .unwrap();
    outcome
        .validate_against_current_cohort(&source, &request, &envelopes)
        .unwrap();
    outcome
        .validate_against_package_current_cohort(&package, &request, &envelopes)
        .unwrap();
    let mut missing = envelopes.clone();
    missing.remove(&request.data_envelope_keys[0]);
    assert!(matches!(
        outcome.validate_against_package_current_cohort(&package, &request, &missing),
        Err(ReplayCurrentCohortError::Missing(
            ReplayCurrentCohortMissing::EnvelopeAbsent { .. }
        ))
    ));
    for unknown in [false, true] {
        let mut request = request.clone();
        let mut outcome = outcome.clone();
        let mut envelopes = envelopes.clone();
        let key = request.data_envelope_keys.remove(0);
        envelopes.remove(&key);
        if unknown {
            request.data_envelope_keys.push("unknown.input".into());
            request.data_envelope_keys.sort();
            let identity = outcome
                .input_data_identities
                .iter_mut()
                .find(|identity| identity.requirement_key == key)
                .unwrap();
            identity.requirement_key = "unknown.input".into();
            identity.identity_fingerprint = identity.compute_fingerprint().unwrap();
            outcome
                .input_data_identities
                .sort_by(|a, b| a.requirement_key.cmp(&b.requirement_key));
        } else {
            outcome
                .input_data_identities
                .retain(|i| i.requirement_key != key);
        }
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        outcome.replay_request_fingerprint = request.request_fingerprint.clone();
        outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
        // Synthetic stored metadata mutant of a genuine three-key replay.
        // Intrinsic validity makes the contextual bundle-coverage refusal meaningful.
        outcome.validate().unwrap();
        request.validate().unwrap();
        let expected = if unknown {
            "unknown data envelope key"
        } else {
            "missing data envelope key"
        };
        assert!(outcome
            .validate_against(&source, &request)
            .unwrap_err()
            .to_string()
            .contains(expected));
        assert!(outcome
            .validate_against_package(&package, &request)
            .unwrap_err()
            .to_string()
            .contains(expected));
        assert!(matches!(
            outcome.validate_against_current_cohort(&source, &request, &envelopes),
            Err(ReplayCurrentCohortError::Invalid(_))
        ));
        assert!(matches!(
            outcome.validate_against_package_current_cohort(&package, &request, &envelopes),
            Err(ReplayCurrentCohortError::Invalid(_))
        ));
    }
}
