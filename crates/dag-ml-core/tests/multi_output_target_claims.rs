//! Genuine public-producer controls using a SYNTHETIC HOST/MOCK, not native/IO science.
#[path = "support/multi_output_mock.rs"]
mod host;
use dag_ml_core::*;
use std::collections::BTreeMap;
use std::sync::{atomic::Ordering, Arc};

fn roots(explain: bool) -> (TrainingOutcome, PortablePredictorPackage) {
    let (source, package) = if explain {
        (
            include_str!("fixtures/multi_output_target_claims/explain/outcome.json"),
            include_str!("fixtures/multi_output_target_claims/explain/package.json"),
        )
    } else {
        (
            include_str!("fixtures/multi_output_target_claims/shared/outcome.json"),
            include_str!("fixtures/multi_output_target_claims/shared/package.json"),
        )
    };
    (
        TrainingOutcome::from_json(source).unwrap(),
        PortablePredictorPackage::from_json(package).unwrap(),
    )
}
fn envelopes(names: Option<&[&str]>) -> BTreeMap<String, ExternalDataPlanEnvelope> {
    let mut values: BTreeMap<String, ExternalDataPlanEnvelope> = serde_json::from_str(
        include_str!("fixtures/multi_output_target_claims/shared/envelopes.json"),
    )
    .unwrap();
    for e in values.values_mut() {
        let old = e.predict_cohort.take().unwrap();
        match names {
            Some(names) => {
                e.predict_cohort = Some(
                    PredictCohort::from_relations(
                        PredictCohortRole::Inference,
                        old.relations,
                        names.iter().map(|s| s.to_string()).collect(),
                        old.data_content_fingerprint,
                        None,
                    )
                    .unwrap(),
                );
            }
            None => {
                e.schema_version = 1;
            }
        }
        e.validate().unwrap();
    }
    values
}
fn request(source: &TrainingOutcome, phase: Phase) -> TrainingReplayRequest {
    let mut r: TrainingReplayRequest = serde_json::from_str(include_str!(
        "fixtures/multi_output_target_claims/shared/request.json"
    ))
    .unwrap();
    r.source_outcome_fingerprint = source.outcome_fingerprint.clone();
    r.phase = phase;
    r.request_fingerprint = r.compute_fingerprint().unwrap();
    r.validate().unwrap();
    r
}
fn execute(
    source: &TrainingOutcome,
    package: &PortablePredictorPackage,
    request: &TrainingReplayRequest,
    envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    attached: bool,
    bad: bool,
    counts: &Arc<host::Counts>,
) -> Result<TrainingReplayOutcome> {
    let controllers = host::registry(package, envelopes, counts, bad);
    let provider = host::Provider {
        envelopes: envelopes.clone(),
        counts: counts.clone(),
    };
    let loaded = LoadedPredictor::<HandleRef>::new(package.clone(), BTreeMap::new()).unwrap();
    if attached {
        execute_attached_training_replay(AttachedTrainingReplayInput {
            source,
            request,
            outcome_id: "outcome:synthetic.attached".into(),
            run_id: RunId::new("run:synthetic.attached").unwrap(),
            controllers: &controllers,
            data_provider: &provider,
            artifact_store: &InMemoryArtifactStore::new(),
            data_envelopes: envelopes,
            warnings: vec![],
            diagnostics: BTreeMap::new(),
        })
    } else {
        execute_loaded_predictor_replay(LoadedPredictorReplayInput {
            predictor: &loaded,
            request,
            outcome_id: "outcome:synthetic.loaded".into(),
            run_id: RunId::new("run:synthetic.loaded").unwrap(),
            controllers: &controllers,
            data_provider: &provider,
            data_envelopes: envelopes,
            warnings: vec![],
            diagnostics: BTreeMap::new(),
        })
    }
}
#[test]
fn genuine_shared_input_unions_predict_and_explain_execute_with_saved_output_names() {
    for phase in [Phase::Predict, Phase::Explain] {
        let (source, package) = roots(phase == Phase::Explain);
        let req = request(&source, phase);
        for names in [["protein", "fat"], ["fat", "protein"]] {
            let env = envelopes(Some(&names));
            preflight_loaded_predictor_replay(&package, &req, &env).unwrap();
            for attached in [true, false] {
                let counts = Arc::new(host::Counts::default());
                let result =
                    execute(&source, &package, &req, &env, attached, false, &counts).unwrap();
                result.validate_against(&source, &req).unwrap();
                result.validate_against_package(&package, &req).unwrap();
                if phase == Phase::Predict {
                    assert_eq!(result.outputs.len(), 2);
                    for output in &result.outputs {
                        assert_eq!(
                            output.predictions[0].target_names,
                            output.binding.target_names
                        );
                    }
                } else {
                    assert_eq!(result.explanations.len(), 2);
                    assert!(result.outputs.is_empty());
                }
                assert_eq!(counts.controller.load(Ordering::SeqCst), 3);
                assert_eq!(counts.hydrate.load(Ordering::SeqCst), 2);
                assert_eq!(counts.release.load(Ordering::SeqCst), 2);
            }
        }
    }
}
#[test]
fn genuine_shared_claim_refusals_precede_all_callbacks_and_late_names_still_refuse() {
    for phase in [Phase::Predict, Phase::Explain] {
        let (source, package) = roots(phase == Phase::Explain);
        let req = request(&source, phase);
        for names in [
            vec!["protein"],
            vec!["fat"],
            vec!["other"],
            vec!["protein", "fat", "moisture"],
        ] {
            let env = envelopes(Some(&names));
            assert!(preflight_loaded_predictor_replay(&package, &req, &env)
                .unwrap_err()
                .to_string()
                .contains("coverage"));
            // Attached has no separate public preflight API. Its pre-callback
            // guard is measured through the actual public execution entrypoint.
            for attached in [true, false] {
                let counts = Arc::new(host::Counts::default());
                assert!(
                    execute(&source, &package, &req, &env, attached, false, &counts)
                        .unwrap_err()
                        .to_string()
                        .contains("coverage")
                );
                counts.zero();
            }
        }
        // EXPLAIN uses honest ExplanationBlocks. Wrong explanation target
        // handling is a separately recorded existing owner behavior, not this
        // prediction-name invariant or the grouped input-claim correction.
        for attached in [true, false]
            .into_iter()
            .filter(|_| phase == Phase::Predict)
        {
            let counts = Arc::new(host::Counts::default());
            let env = envelopes(Some(&["protein", "fat"]));
            preflight_loaded_predictor_replay(&package, &req, &env).unwrap();
            let error = execute(&source, &package, &req, &env, attached, true, &counts)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(if phase == Phase::Predict {
                    "target order"
                } else {
                    "target"
                }),
                "{error}"
            );
            assert_eq!(counts.controller.load(Ordering::SeqCst), 3);
            assert_eq!(counts.release.load(Ordering::SeqCst), 2);
        }
        let env = envelopes(None);
        let counts = Arc::new(host::Counts::default());
        execute(&source, &package, &req, &env, false, false, &counts).unwrap();
    }
}
struct Cohort(BTreeMap<String, ExternalDataPlanEnvelope>);
impl FacetReplayCohort for Cohort {
    fn envelopes(&self) -> &BTreeMap<String, ExternalDataPlanEnvelope> {
        &self.0
    }
}
#[test]
fn canonical_facet_admission_preserves_union_coverage_order_and_presence_types() {
    let (source, package) = roots(false);
    let req = request(&source, Phase::Predict);
    let export = build_training_artifact_facets_from_bytes(
        &serde_json::to_vec(&source).unwrap(),
        &serde_json::to_vec(&package).unwrap(),
    )
    .unwrap();
    for names in [["protein", "fat"], ["fat", "protein"]] {
        admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(req.clone()),
            Cohort(envelopes(Some(&names))),
        )
        .unwrap();
    }
    assert!(matches!(
        admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(req.clone()),
            Cohort(envelopes(Some(&["protein"])))
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(req.clone()),
            Cohort(envelopes(None))
        ),
        Err(FacetRefusal::UnverifiedEvidence { .. })
    ));
    let equal_source = include_bytes!("fixtures/multi_output_target_claims/equal/outcome.json");
    let equal_package = include_bytes!("fixtures/multi_output_target_claims/equal/package.json");
    let equal = build_training_artifact_facets_from_bytes(equal_source, equal_package).unwrap();
    let equal_source =
        TrainingOutcome::from_json(std::str::from_utf8(equal_source).unwrap()).unwrap();
    let req = request(&equal_source, Phase::Predict);
    assert!(matches!(
        admit_facet_predict(
            &equal.graph,
            &equal.context,
            &equal.members,
            "model",
            FacetPredictRequest::Replay(req),
            Cohort(envelopes(Some(&["fat", "protein"])))
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
    let req = request(&source, Phase::Explain);
    assert!(matches!(
        admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(req),
            Cohort(envelopes(Some(&["protein", "fat"])))
        ),
        Err(FacetRefusal::UnsupportedOperation { .. })
    ));
}
