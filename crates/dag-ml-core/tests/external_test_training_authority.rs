//! Owner authority controls over the saved b1 and V3 packages.
use dag_ml_core::*;
use std::collections::BTreeMap;

#[path = "support/multi_output_mock.rs"]
#[allow(dead_code)]
mod synthetic_host;

const PACKAGE: &str = include_str!("fixtures/artifact_facets_b1/outputs/ppc.before.json");
const OUTCOME: &str =
    include_str!("fixtures/artifact_facets_b1/outputs/training_outcome.before.json");
const REQUEST: &str = include_str!("fixtures/artifact_facets_b1/outputs/replay_request.json");
const ENVELOPES: &str = include_str!("fixtures/artifact_facets_b1/outputs/replay_envelope.json");
const CHILD: &str = include_str!("fixtures/artifact_facets/native_role/child_refit_v3.json");
const TARGET_RELATIONS: &str =
    include_str!("fixtures/artifact_facets/native_role/target_relations.json");

fn inputs() -> (
    PortablePredictorPackage,
    TrainingReplayRequest,
    BTreeMap<String, ExternalDataPlanEnvelope>,
) {
    (
        PortablePredictorPackage::from_json(PACKAGE).unwrap(),
        TrainingReplayRequest::from_json(REQUEST).unwrap(),
        serde_json::from_str(ENVELOPES).unwrap(),
    )
}

fn unrelated() -> SampleRelationSet {
    SampleRelationSet {
        records: vec![SampleRelation::new(
            ObservationId::new("observation:authority:unrelated").unwrap(),
            SampleId::new("sample:authority:unrelated").unwrap(),
        )],
    }
}

fn wrong_set(envelope: &mut ExternalDataPlanEnvelope, matching_string: bool) {
    let relations = unrelated();
    if matching_string {
        envelope.relation_fingerprint = Some(relations.fingerprint().unwrap());
    }
    envelope.coordinator_relations = Some(relations);
    envelope.validate().unwrap();
}

#[test]
fn b1_identity_closure_and_fingerprint_precedence() {
    let (package, _, envelopes) = inputs();
    let relations = envelopes
        .values()
        .next()
        .unwrap()
        .coordinator_relations
        .as_ref()
        .unwrap();
    assert_eq!(relations.records.len(), 40);
    package
        .training_influence
        .validate_identity_closure(relations)
        .unwrap();
    let error = package
        .training_influence
        .validate_identity_closure(&unrelated())
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("relation fingerprint does not match"));
}

#[test]
fn stale_influence_origins_groups_and_samples_are_refused_with_bound_hash() {
    let (package, _, envelopes) = inputs();
    let relations = envelopes
        .values()
        .next()
        .unwrap()
        .coordinator_relations
        .as_ref()
        .unwrap();
    for field in ["origin", "group", "sample"] {
        let mut manifest = package.training_influence.clone();
        let entry = &mut manifest.entries[0];
        match field {
            "origin" => {
                entry.origin_sample_ids = vec![SampleId::new("sample:stale:origin").unwrap()]
            }
            "group" => entry.group_ids = vec![GroupId::new("group:stale").unwrap()],
            _ => {
                entry.physical_sample_ids = vec![SampleId::new("sample:stale:physical").unwrap()];
            }
        }
        manifest.manifest_fingerprint = manifest.compute_fingerprint().unwrap();
        manifest.validate().unwrap();
        let error = manifest
            .validate_identity_closure(relations)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(match field {
                "origin" => "origin closure",
                "group" => "group closure",
                _ => "absent from relation set",
            }),
            "{field}: {error}"
        );
    }
}

#[test]
fn identity_and_projection_agree_on_stale_child_closure() {
    let child = PortableRefitPackageV3::from_json(CHILD).unwrap();
    let relations: SampleRelationSet = serde_json::from_str(TARGET_RELATIONS).unwrap();
    let request = &child.outcome.target_training_request;
    let projection = request.project().unwrap();
    let manifest = &child.outcome.training_influence;
    manifest.validate_identity_closure(&relations).unwrap();
    manifest
        .validate_for_projection(&projection, request, &relations)
        .unwrap();
    for field in ["origin", "group", "sample"] {
        let mut stale = manifest.clone();
        let mut test_relations = relations.clone();
        let mut test_request = request.clone();
        match field {
            "origin" => {
                stale.entries[0].origin_sample_ids =
                    vec![SampleId::new("sample:stale:origin").unwrap()]
            }
            "group" => stale.entries[0].group_ids = vec![GroupId::new("group:stale").unwrap()],
            _ => {
                // Preserve the projection's scope so both validators reach identity closure.
                let missing_sample = &stale.entries[0].physical_sample_ids[0];
                test_relations
                    .records
                    .retain(|relation| &relation.sample_id != missing_sample);
                let fingerprint = test_relations.fingerprint().unwrap();
                stale.relation_fingerprint = fingerprint.clone();
                for identity in &mut test_request.data_identities {
                    identity.relation_fingerprint = fingerprint.clone();
                    identity.identity_fingerprint = identity.compute_fingerprint().unwrap();
                    identity.validate().unwrap();
                }
                test_request.request_fingerprint = test_request.compute_fingerprint().unwrap();
            }
        }
        stale.manifest_fingerprint = stale.compute_fingerprint().unwrap();
        stale.validate().unwrap();
        let expected = DagMlError::CampaignValidation(format!(
            "training influence `{}` {}",
            stale.entries[0].scope_id,
            match field {
                "origin" => "origin closure does not match relation set",
                "group" => "group closure does not match relation set",
                _ => "contains physical samples absent from relation set",
            }
        ))
        .to_string();
        assert_eq!(
            stale
                .validate_identity_closure(&test_relations)
                .unwrap_err()
                .to_string(),
            expected,
            "{field} identity closure"
        );
        assert_eq!(
            stale
                .validate_for_projection(&projection, &test_request, &test_relations)
                .unwrap_err()
                .to_string(),
            expected,
            "{field} projection closure"
        );
    }
}

#[test]
fn loaded_and_facet_admission_refuse_unbound_training_relations() {
    struct Cohort(BTreeMap<String, ExternalDataPlanEnvelope>);
    impl FacetReplayCohort for Cohort {
        fn envelopes(&self) -> &BTreeMap<String, ExternalDataPlanEnvelope> {
            &self.0
        }
    }
    let (package, request, envelopes) = inputs();
    preflight_loaded_predictor_replay(&package, &request, &envelopes).unwrap();
    let export =
        build_training_artifact_facets_from_bytes(OUTCOME.as_bytes(), PACKAGE.as_bytes()).unwrap();
    for matching_string in [true, false] {
        let mut bad = envelopes.clone();
        for envelope in bad.values_mut() {
            wrong_set(envelope, matching_string);
        }
        for phase in [Phase::Predict, Phase::Explain] {
            let mut current = request.clone();
            current.phase = phase;
            current.request_fingerprint = current.compute_fingerprint().unwrap();
            let error = preflight_loaded_predictor_replay(&package, &current, &bad).unwrap_err();
            assert!(error
                .to_string()
                .contains("external_test training relations do not match owner authority"));
        }
        let refusal = match admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(request.clone()),
            Cohort(bad),
        ) {
            Ok(_) => panic!("facet admitted unbound training relations"),
            Err(refusal) => refusal,
        };
        assert!(
            matches!(&refusal, FacetRefusal::OwnerValidation { source, .. }
            if source.to_string().contains("external_test training relations do not match owner authority"))
        );
    }
    for fingerprint in [None, Some("a".repeat(64))] {
        let mut bad = envelopes.clone();
        for envelope in bad.values_mut() {
            envelope.relation_fingerprint = fingerprint.clone();
        }
        let refusal = match admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(request.clone()),
            Cohort(bad.clone()),
        ) {
            Ok(_) => panic!("facet admitted an unbound fingerprint"),
            Err(refusal) => refusal,
        };
        assert!(
            matches!(&refusal, FacetRefusal::OwnerValidation { source, .. }
            if source.to_string().contains("external_test training relations do not match owner authority"))
        );
        let mut explain = request.clone();
        explain.phase = Phase::Explain;
        explain.request_fingerprint = explain.compute_fingerprint().unwrap();
        assert!(matches!(
            admit_facet_predict(
                &export.graph,
                &export.context,
                &export.members,
                "model",
                FacetPredictRequest::Replay(explain),
                Cohort(bad),
            ),
            Err(FacetRefusal::UnsupportedOperation { .. })
        ));
    }
}

#[test]
fn missing_or_wrong_authority_is_refused_and_non_external_roles_remain_unchanged() {
    let (package, request, envelopes) = inputs();
    let original = envelopes.values().next().unwrap();
    let expected = original.relation_fingerprint.as_deref().unwrap();
    let folds = package.effective_plan.fold_set.as_ref();
    original
        .validate_external_test_training_authority(Some(expected), Some(expected), folds)
        .unwrap();
    assert!(original
        .validate_external_test_training_authority(None, Some(expected), folds)
        .is_err());
    let mut bad = original.clone();
    bad.coordinator_relations = None;
    assert!(bad
        .validate_external_test_training_authority(Some(expected), Some(expected), folds)
        .is_err());
    let mut missing_map = envelopes.clone();
    *missing_map.values_mut().next().unwrap() = bad;
    assert!(preflight_loaded_predictor_replay(&package, &request, &missing_map).is_err());
    for fingerprint in [None, Some("a".repeat(64))] {
        let mut bad = original.clone();
        bad.relation_fingerprint = fingerprint;
        assert!(bad
            .validate_external_test_training_authority(Some(expected), Some(expected), folds)
            .is_err());
        let mut bad_map = envelopes.clone();
        *bad_map.values_mut().next().unwrap() = bad;
        assert!(
            preflight_loaded_predictor_replay(&package, &request, &bad_map)
                .unwrap_err()
                .to_string()
                .contains("external_test training relations do not match owner authority")
        );
    }
    let mut v1 = envelopes.clone();
    for envelope in v1.values_mut() {
        envelope.schema_version = 1;
        envelope.predict_cohort = None;
    }
    preflight_loaded_predictor_replay(&package, &request, &v1).unwrap();
    let mut inference = envelopes.clone();
    for envelope in inference.values_mut() {
        let cohort = envelope.predict_cohort.as_mut().unwrap();
        cohort.role = PredictCohortRole::Inference;
        cohort.target_content_fingerprint = None;
        cohort.cohort_fingerprint = cohort.fingerprint().unwrap();
        envelope.target_content_fingerprint = None;
        envelope.coordinator_relations = Some(cohort.relations.clone());
        envelope.relation_fingerprint = Some(cohort.relations.fingerprint().unwrap());
    }
    preflight_loaded_predictor_replay(&package, &request, &inference).unwrap();
}

#[test]
fn external_cohort_cannot_reuse_training_origin_alias() {
    let (_, _, envelopes) = inputs();
    let mut envelope = envelopes.values().next().unwrap().clone();
    let cohort_id = envelope
        .predict_cohort
        .as_ref()
        .unwrap()
        .physical_sample_ids[0]
        .clone();
    let mut relations = unrelated();
    relations.records[0].origin_sample_id = Some(cohort_id);
    let fingerprint = relations.fingerprint().unwrap();
    envelope.relation_fingerprint = Some(fingerprint.clone());
    envelope.coordinator_relations = Some(relations);
    let error = envelope
        .validate_external_test_training_authority(Some(&fingerprint), Some(&fingerprint), None)
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("overlaps CV relation identity closure"));
}

#[test]
fn cv_fold_sample_origin_and_target_missing_are_refused() {
    let (package, request, envelopes) = inputs();
    let original = envelopes.values().next().unwrap();
    let expected = original.relation_fingerprint.as_deref().unwrap();
    let mut folds = package.effective_plan.fold_set.clone().unwrap();
    let mut cohort = original.predict_cohort.clone().unwrap();
    let missing = cohort.physical_sample_ids[0].clone();
    folds.target_missing_sample_ids.push(missing);
    folds.validate().unwrap();
    let mut envelope = original.clone();
    assert!(envelope
        .validate_external_test_training_authority(Some(expected), Some(expected), Some(&folds))
        .unwrap_err()
        .to_string()
        .contains("target-missing"));
    folds.target_missing_sample_ids.clear();
    for origin in [false, true] {
        let mut relations = unrelated();
        if origin {
            relations.records[0].origin_sample_id = Some(folds.sample_ids[0].clone());
        } else {
            relations.records[0].sample_id = folds.sample_ids[0].clone();
        }
        cohort = PredictCohort::from_relations(
            PredictCohortRole::ExternalTest,
            relations,
            cohort.target_names.clone(),
            cohort.data_content_fingerprint.clone(),
            cohort.target_content_fingerprint.clone(),
        )
        .unwrap();
        envelope.predict_cohort = Some(cohort.clone());
        assert_eq!(
            envelope
                .validate_external_test_training_authority(
                    Some(expected),
                    Some(expected),
                    Some(&folds)
                )
                .unwrap_err()
                .to_string(),
            DagMlError::CampaignValidation(format!(
                "external_test predict cohort overlaps CV fold sample or origin `{}`",
                folds.sample_ids[0]
            ))
            .to_string()
        );
        // The real replay path first performs generic envelope validation;
        // the genuine b1 relation universe already contains this fold ID.
        let mut replay_envelopes = envelopes.clone();
        *replay_envelopes.values_mut().next().unwrap() = envelope.clone();
        for phase in [Phase::Predict, Phase::Explain] {
            let mut current = request.clone();
            current.phase = phase;
            current.request_fingerprint = current.compute_fingerprint().unwrap();
            let error = preflight_loaded_predictor_replay(&package, &current, &replay_envelopes)
                .unwrap_err()
                .to_string();
            assert_eq!(
                error,
                DagMlError::CampaignValidation(format!(
                    "external_test predict cohort overlaps CV relation identity closure `{}`",
                    folds.sample_ids[0]
                ))
                .to_string(),
                "{phase:?}"
            );
        }
    }
}

#[test]
fn v3_actual_child_runtime_requirements_equal_target_relation() {
    let child = PortableRefitPackageV3::from_json(CHILD).unwrap();
    let bundle = child.outcome.to_runtime_replay_bundle().unwrap();
    assert!(!bundle.data_requirements.is_empty());
    for requirement in &bundle.data_requirements {
        assert_eq!(
            requirement.relation_fingerprint.as_deref(),
            Some(
                child
                    .outcome
                    .provenance
                    .target_relation_fingerprint
                    .as_str()
            )
        );
    }
    let (_, mut request, prototype) = inputs();
    let relations: SampleRelationSet = serde_json::from_str(TARGET_RELATIONS).unwrap();
    let template = prototype.values().next().unwrap();
    let envelopes: BTreeMap<_, _> = bundle
        .data_requirements
        .iter()
        .map(|requirement| {
            let mut envelope = template.clone();
            envelope.schema_fingerprint = requirement.schema_fingerprint.clone();
            envelope.plan_fingerprint = requirement.plan_fingerprint.clone();
            envelope.relation_fingerprint = Some(relations.fingerprint().unwrap());
            envelope.coordinator_relations = Some(relations.clone());
            (requirement.key(), envelope)
        })
        .collect();
    request.source_outcome_fingerprint = child.outcome.outcome_fingerprint.clone();
    request.data_envelope_keys = envelopes.keys().cloned().collect();
    request.output_binding_ids = child
        .outcome
        .output_bindings
        .iter()
        .map(|binding| binding.binding_id.clone())
        .collect();
    request.output_binding_ids.sort();
    request.request_fingerprint = request.compute_fingerprint().unwrap();
    preflight_loaded_portable_refit_replay_v3(&child, &request, &envelopes).unwrap();
    for mutation in 0..4 {
        let mut bad = envelopes.clone();
        for envelope in bad.values_mut() {
            match mutation {
                0 => wrong_set(envelope, true),
                1 => wrong_set(envelope, false),
                2 => envelope.relation_fingerprint = None,
                _ => envelope.relation_fingerprint = Some("a".repeat(64)),
            }
        }
        for phase in [Phase::Predict, Phase::Explain] {
            let mut current = request.clone();
            current.phase = phase;
            current.request_fingerprint = current.compute_fingerprint().unwrap();
            let error =
                preflight_loaded_portable_refit_replay_v3(&child, &current, &bad).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("external_test training relations do not match owner authority"),
                "mutation {mutation}, {phase:?}: {error}"
            );
        }
    }
}

#[test]
fn initial_refit_uses_its_package_training_relations() {
    let package = InitialFullRefitPackage::from_json(include_str!(
        "fixtures/initial_full_refit/package.json"
    ))
    .unwrap();
    let cohort = PredictCohort::from_relations(
        PredictCohortRole::ExternalTest,
        unrelated(),
        vec!["target:y".to_string()],
        "c".repeat(64),
        Some("d".repeat(64)),
    )
    .unwrap();
    let envelope = package.predict_envelope(cohort).unwrap();
    let outputs: Vec<_> = package
        .outputs
        .iter()
        .map(|output| output.output_id.clone())
        .collect();
    preflight_initial_full_refit_prediction(&package, &envelope, &outputs).unwrap();
    struct Cohort(BTreeMap<String, ExternalDataPlanEnvelope>);
    impl FacetReplayCohort for Cohort {
        fn envelopes(&self) -> &BTreeMap<String, ExternalDataPlanEnvelope> {
            &self.0
        }
    }
    let export = build_initial_refit_artifact_facets(&package).unwrap();
    let facet_cohort = |envelope: &ExternalDataPlanEnvelope| {
        Cohort(
            package
                .effective_plan
                .node_plans
                .values()
                .flat_map(|node| &node.data_bindings)
                .map(|binding| {
                    (
                        data_binding_requirement_key(&binding.node_id, &binding.input_name),
                        envelope.clone(),
                    )
                })
                .collect(),
        )
    };
    for mutation in 0..4 {
        let mut bad = envelope.clone();
        match mutation {
            0 | 1 => {
                let relations = SampleRelationSet {
                    records: vec![SampleRelation::new(
                        ObservationId::new("observation:authority:other").unwrap(),
                        SampleId::new("sample:authority:other").unwrap(),
                    )],
                };
                if mutation == 0 {
                    bad.relation_fingerprint = Some(relations.fingerprint().unwrap());
                }
                bad.coordinator_relations = Some(relations);
            }
            2 => bad.relation_fingerprint = None,
            _ => bad.relation_fingerprint = Some("a".repeat(64)),
        }
        let result = admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "full_refit",
            FacetPredictRequest::InitialRefit {
                output_ids: outputs.clone(),
            },
            facet_cohort(&bad),
        );
        assert!(
            matches!(result, Err(FacetRefusal::OwnerValidation { source, .. })
            if source.to_string().contains("external_test training relations do not match owner authority"))
        );
    }
    for fingerprint in [None, Some("a".repeat(64))] {
        let mut bad = envelope.clone();
        bad.relation_fingerprint = fingerprint;
        let error = preflight_initial_full_refit_prediction(&package, &bad, &outputs).unwrap_err();
        assert!(error
            .to_string()
            .contains("external_test training relations do not match owner authority"));
    }
    let mut bad = envelope.clone();
    let mut wrong = package.training_relations.clone();
    wrong.records[0]
        .metadata
        .insert("authority_test".into(), serde_json::json!("changed"));
    bad.coordinator_relations = Some(wrong);
    bad.validate().unwrap();
    let error = preflight_initial_full_refit_prediction(&package, &bad, &outputs).unwrap_err();
    assert!(error
        .to_string()
        .contains("external_test training relations do not match owner authority"));
    bad.relation_fingerprint = Some(
        bad.coordinator_relations
            .as_ref()
            .unwrap()
            .fingerprint()
            .unwrap(),
    );
    bad.validate().unwrap();
    let error = preflight_initial_full_refit_prediction(&package, &bad, &outputs).unwrap_err();
    assert!(error
        .to_string()
        .contains("external_test training relations do not match owner authority"));
}

// Relation-only snapshot of the real DMD producer, not a host-made S=C packet.
fn real_dmd_initial_package() -> (InitialFullRefitPackage, serde_json::Value) {
    let producer: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/initial_full_refit/real_dmd_train_relations.json"
    ))
    .unwrap();
    let mut package = InitialFullRefitPackage::from_json(include_str!(
        "fixtures/initial_full_refit/package.json"
    ))
    .unwrap();
    package.training_relations =
        serde_json::from_value(producer["coordinator_relations"].clone()).unwrap();
    package.training_sample_ids =
        serde_json::from_value(producer["training_sample_ids"].clone()).unwrap();
    package.training_envelope.coordinator_relations = Some(package.training_relations.clone());
    package.training_envelope.relation_fingerprint =
        Some(producer["source_key"].as_str().unwrap().into());
    let mut campaign = package.effective_plan.campaign.clone();
    for binding in campaign.data_bindings.values_mut().flatten() {
        binding.relation_fingerprint = package.training_envelope.relation_fingerprint.clone();
    }
    let mut manifests = ControllerRegistry::new();
    for manifest in package.effective_plan.controller_manifests.values() {
        manifests.register(manifest.clone()).unwrap();
    }
    package.effective_plan = build_execution_plan(
        "plan:real.dmd.initial",
        package.effective_plan.graph_plan.graph.clone(),
        campaign,
        &manifests,
    )
    .unwrap();
    package.effective_plan_fingerprint =
        parse_typed_json(&serde_json::to_string(&package.effective_plan).unwrap())
            .unwrap()
            .fingerprint()
            .unwrap();
    package.variant_id = package.effective_plan.variants[0].variant_id.clone();
    package.data_identities = package
        .effective_plan
        .node_plans
        .values()
        .flat_map(|node| &node.data_bindings)
        .map(|binding| {
            TrainingDataIdentity::from_binding_envelope(binding, &package.training_envelope)
                .unwrap()
        })
        .collect();
    package.package_fingerprint = package.compute_fingerprint().unwrap();
    package.validate().unwrap();
    (package, producer)
}

#[test]
fn real_dmd_initial_package_round_trips_and_replays_with_distinct_domains() {
    let (package, producer) = real_dmd_initial_package();
    let canonical = package.training_relations.fingerprint().unwrap();
    let source = producer["source_key"].as_str().unwrap();
    assert_eq!(canonical, producer["canonical_key"].as_str().unwrap());
    assert_ne!(source, canonical);
    let roundtrip =
        InitialFullRefitPackage::from_json(&serde_json::to_string(&package).unwrap()).unwrap();
    assert_eq!(roundtrip, package);
    let cohort = PredictCohort::from_relations(
        PredictCohortRole::ExternalTest,
        unrelated(),
        vec!["y".into()],
        "c".repeat(64),
        Some("d".repeat(64)),
    )
    .unwrap();
    let envelope = package.predict_envelope(cohort).unwrap();
    let outputs: Vec<_> = package
        .outputs
        .iter()
        .map(|o| o.output_id.clone())
        .collect();
    envelope
        .validate_external_test_training_authority(Some(source), Some(&canonical), None)
        .unwrap();
    preflight_initial_full_refit_prediction(&package, &envelope, &outputs).unwrap();
    let selector = TerminalPredictionSelector::new(
        package.outputs[0].node_id.clone(),
        package.outputs[0].port_name.clone(),
    )
    .unwrap();
    // The plan owns S only; it does not invent a canonical key from S.
    validate_terminal_prediction_preflight(&package.effective_plan, &envelope, &selector).unwrap();
    let mut bad_source = envelope.clone();
    bad_source.relation_fingerprint = Some("f".repeat(64));
    assert!(preflight_initial_full_refit_prediction(&package, &bad_source, &outputs).is_err());
    assert!(validate_terminal_prediction_preflight(
        &package.effective_plan,
        &bad_source,
        &selector
    )
    .is_err());
    let mut bad_records = envelope.clone();
    bad_records.coordinator_relations.as_mut().unwrap().records[0]
        .metadata
        .insert("changed".into(), serde_json::json!(true));
    assert!(preflight_initial_full_refit_prediction(&package, &bad_records, &outputs).is_err());
    for change_identity in [false, true] {
        let mut forged = package.clone();
        if change_identity {
            forged.data_identities[0].relation_fingerprint = "f".repeat(64);
            forged.data_identities[0].identity_fingerprint =
                forged.data_identities[0].compute_fingerprint().unwrap();
        } else {
            forged.training_relations.records[0]
                .metadata
                .insert("changed".into(), serde_json::json!(true));
        }
        forged.package_fingerprint = forged.compute_fingerprint().unwrap();
        assert!(forged.validate().is_err());
    }
    let overlap = PredictCohort::from_relations(
        PredictCohortRole::ExternalTest,
        package.training_relations.clone(),
        vec!["y".into()],
        "c".repeat(64),
        Some("d".repeat(64)),
    )
    .unwrap();
    assert!(package.predict_envelope(overlap).is_err());
}

#[test]
fn real_dmd_initial_refit_refuses_provider_records_identity_and_order_before_callbacks() {
    struct Provider {
        records: SampleRelationSet,
        identity: TrainingDataIdentity,
        ids: Vec<SampleId>,
    }
    impl RuntimeDataProvider for Provider {
        fn materialize(&self, _: &DataMaterializationRequest) -> Result<HandleRef> {
            panic!("invalid attestation must be rejected before materialization");
        }
        fn make_view(&self, _: &DataViewRequest) -> Result<HandleRef> {
            panic!("invalid attestation must be rejected before view creation");
        }
        fn training_data_identity(&self, _: &DataBinding) -> Result<Option<TrainingDataIdentity>> {
            Ok(Some(self.identity.clone()))
        }
        fn coordinator_relations(&self, _: &DataBinding) -> Result<Option<SampleRelationSet>> {
            Ok(Some(self.records.clone()))
        }
        fn refit_sample_ids(&self, _: &DataBinding) -> Result<Option<Vec<SampleId>>> {
            Ok(Some(self.ids.clone()))
        }
    }
    let (package, _) = real_dmd_initial_package();
    for mutation in 0..6 {
        let mut provider = Provider {
            records: package.training_relations.clone(),
            identity: package.data_identities[0].clone(),
            ids: package.training_sample_ids.clone(),
        };
        match mutation {
            0 => {
                provider.records.records[0]
                    .metadata
                    .insert("changed".into(), serde_json::json!(true));
            }
            1 => provider.records.records[0].group_id = Some(GroupId::new("changed").unwrap()),
            2 => {
                provider.records.records[0].origin_sample_id =
                    Some(package.training_sample_ids[0].clone());
            }
            3 => provider.records.records[0].excluded = true,
            4 => {
                provider.identity.target_content_fingerprint = "a".repeat(64);
                assert_ne!(
                    provider.identity.target_content_fingerprint,
                    package.data_identities[0].target_content_fingerprint
                );
                provider.identity.identity_fingerprint =
                    provider.identity.compute_fingerprint().unwrap();
            }
            _ => provider.ids.swap(0, 1),
        }
        let controllers = RuntimeControllerRegistry::new();
        let error = execute_initial_full_refit(InitialFullRefitExecutionInput {
            package_id: "package:real.dmd.refused".into(),
            run_id: RunId::new("run:real.dmd.refused").unwrap(),
            plan: &package.effective_plan,
            training_envelope: &package.training_envelope,
            training_sample_ids: &package.training_sample_ids,
            controllers: &controllers,
            data_provider: &provider,
            root_seed: Some(0),
            resource_limits: None,
            scheduler: InitialRefitScheduler::Sequential,
        })
        .err()
        .expect("mismatched native provider attestation must be refused");
        assert!(
            error.to_string().contains(match mutation {
                0..=3 => "provider relations differ",
                4 => "provider identity does not match",
                _ => "training order differs",
            }),
            "mutation {mutation}: {error}"
        );
    }
}

#[test]
fn generic_sample12_validation_remains_available() {
    let mut envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
        "fixtures/package/data/coordinator_data_plan_envelope_sample12.json"
    ))
    .unwrap();
    let cohort = PredictCohort::from_relations(
        PredictCohortRole::ExternalTest,
        unrelated(),
        vec!["target:y".to_string()],
        "c".repeat(64),
        Some("d".repeat(64)),
    )
    .unwrap();
    envelope.schema_version = 2;
    envelope.predict_cohort = Some(cohort);
    envelope.validate().unwrap();
}

#[test]
fn v1_fresh_cohort_replay_uses_new_relation_fingerprint() {
    use std::sync::{atomic::Ordering, Arc};
    // Existing synthetic producer and host: this exercises replay plumbing,
    // not native numerical execution or native byte equivalence.
    let source = TrainingOutcome::from_json(include_str!(
        "fixtures/multi_output_target_claims/shared/outcome.json"
    ))
    .unwrap();
    let package = PortablePredictorPackage::from_json(include_str!(
        "fixtures/multi_output_target_claims/shared/package.json"
    ))
    .unwrap();
    let mut request: TrainingReplayRequest = serde_json::from_str(include_str!(
        "fixtures/multi_output_target_claims/shared/request.json"
    ))
    .unwrap();
    request.source_outcome_fingerprint = source.outcome_fingerprint.clone();
    request.phase = Phase::Predict;
    request.request_fingerprint = request.compute_fingerprint().unwrap();
    let mut envelopes: BTreeMap<String, ExternalDataPlanEnvelope> = serde_json::from_str(
        include_str!("fixtures/multi_output_target_claims/shared/envelopes.json"),
    )
    .unwrap();
    for envelope in envelopes.values_mut() {
        envelope.schema_version = 1;
        envelope.predict_cohort = None;
        envelope.validate().unwrap();
    }
    let key = &request.data_envelope_keys[0];
    let fresh = envelopes[key].relation_fingerprint.as_ref().unwrap();
    let original = package
        .execution_bundle
        .data_requirements
        .iter()
        .find(|requirement| requirement.key() == *key)
        .unwrap();
    assert_ne!(original.relation_fingerprint.as_ref(), Some(fresh));
    let counts = Arc::new(synthetic_host::Counts::default());
    let controllers = synthetic_host::registry(&package, &envelopes, &counts, false);
    let provider = synthetic_host::Provider {
        envelopes: envelopes.clone(),
        counts: counts.clone(),
    };
    let loaded = LoadedPredictor::<HandleRef>::new(package.clone(), BTreeMap::new()).unwrap();
    let result = execute_loaded_predictor_replay(LoadedPredictorReplayInput {
        predictor: &loaded,
        request: &request,
        outcome_id: "outcome:authority:v1".into(),
        run_id: RunId::new("run:authority:v1").unwrap(),
        controllers: &controllers,
        data_provider: &provider,
        data_envelopes: &envelopes,
        warnings: Vec::new(),
        diagnostics: BTreeMap::new(),
    })
    .unwrap();
    result.validate_against_package(&package, &request).unwrap();
    assert_eq!(
        &result
            .input_data_identities
            .iter()
            .find(|identity| identity.requirement_key == *key)
            .unwrap()
            .relation_fingerprint,
        fresh
    );
    assert!(counts.controller.load(Ordering::SeqCst) > 0);
}

#[test]
fn attached_and_loaded_execution_refuse_before_callbacks() {
    use std::collections::BTreeSet;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    struct Controller {
        id: ControllerId,
        calls: Arc<AtomicUsize>,
    }
    impl RuntimeController for Controller {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }
        fn invoke(&self, _: &NodeTask) -> Result<NodeResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(DagMlError::RuntimeValidation("unexpected callback".into()))
        }
        fn hydrate_artifact_payload(
            &self,
            _: &ArtifactMaterializationRequest,
            _: &[u8],
        ) -> Result<HandleRef> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(DagMlError::RuntimeValidation("unexpected callback".into()))
        }
    }
    struct Provider(Arc<AtomicUsize>);
    impl RuntimeDataProvider for Provider {
        fn materialize(&self, _: &DataMaterializationRequest) -> Result<HandleRef> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(DagMlError::RuntimeValidation("unexpected callback".into()))
        }
        fn make_view(&self, _: &DataViewRequest) -> Result<HandleRef> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(DagMlError::RuntimeValidation("unexpected callback".into()))
        }
    }
    struct Store(Arc<AtomicUsize>);
    impl RuntimeArtifactStore for Store {
        fn materialize(&self, _: &ArtifactMaterializationRequest) -> Result<HandleRef> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(DagMlError::RuntimeValidation("unexpected callback".into()))
        }
    }
    let source = TrainingOutcome::from_json(OUTCOME).unwrap();
    let (package, request, original_envelopes) = inputs();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut controllers = RuntimeControllerRegistry::new();
    for id in package
        .effective_plan
        .node_plans
        .values()
        .map(|node| node.controller_id.clone())
        .collect::<BTreeSet<_>>()
    {
        controllers
            .register(Box::new(Controller {
                id,
                calls: calls.clone(),
            }))
            .unwrap();
    }
    let provider = Provider(calls.clone());
    let store = Store(calls.clone());
    let loaded = LoadedPredictor::<HandleRef>::new(package, BTreeMap::new()).unwrap();
    for mutation in 0..4 {
        let mut envelopes = original_envelopes.clone();
        for envelope in envelopes.values_mut() {
            match mutation {
                0 => wrong_set(envelope, true),
                1 => wrong_set(envelope, false),
                2 => envelope.relation_fingerprint = None,
                _ => envelope.relation_fingerprint = Some("a".repeat(64)),
            }
            envelope.validate().unwrap();
        }
        for phase in [Phase::Predict, Phase::Explain] {
            let mut current = request.clone();
            current.phase = phase;
            current.request_fingerprint = current.compute_fingerprint().unwrap();
            if phase == Phase::Predict {
                // The saved b1 outcome declares only PREDICT replayable.
                let attached = execute_attached_training_replay(AttachedTrainingReplayInput {
                    source: &source,
                    request: &current,
                    outcome_id: "outcome:authority:attached".into(),
                    run_id: RunId::new("run:authority:attached").unwrap(),
                    controllers: &controllers,
                    data_provider: &provider,
                    artifact_store: &store,
                    data_envelopes: &envelopes,
                    warnings: Vec::new(),
                    diagnostics: BTreeMap::new(),
                })
                .unwrap_err();
                assert!(attached
                    .to_string()
                    .contains("external_test training relations do not match owner authority"));
            }
            let detached = execute_loaded_predictor_replay(LoadedPredictorReplayInput {
                predictor: &loaded,
                request: &current,
                outcome_id: "outcome:authority:loaded".into(),
                run_id: RunId::new("run:authority:loaded").unwrap(),
                controllers: &controllers,
                data_provider: &provider,
                data_envelopes: &envelopes,
                warnings: Vec::new(),
                diagnostics: BTreeMap::new(),
            })
            .unwrap_err();
            assert!(detached
                .to_string()
                .contains("external_test training relations do not match owner authority"));
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }
    }

    // This saved producer artifact really declares EXPLAIN. Its producer is a
    // synthetic host; the callback traps below only measure refusal timing.
    let explain_source = TrainingOutcome::from_json(include_str!(
        "fixtures/multi_output_target_claims/explain/outcome.json"
    ))
    .unwrap();
    assert!(explain_source.replayable_phases.contains(&Phase::Explain));
    let mut explain_request: TrainingReplayRequest = serde_json::from_str(include_str!(
        "fixtures/multi_output_target_claims/shared/request.json"
    ))
    .unwrap();
    explain_request.source_outcome_fingerprint = explain_source.outcome_fingerprint.clone();
    explain_request.phase = Phase::Explain;
    explain_request.request_fingerprint = explain_request.compute_fingerprint().unwrap();
    let mut explain_envelopes: BTreeMap<String, ExternalDataPlanEnvelope> = serde_json::from_str(
        include_str!("fixtures/multi_output_target_claims/shared/envelopes.json"),
    )
    .unwrap();
    for envelope in explain_envelopes.values_mut() {
        let old = envelope.predict_cohort.take().unwrap();
        envelope.predict_cohort = Some(
            PredictCohort::from_relations(
                PredictCohortRole::ExternalTest,
                old.relations,
                vec!["protein".into(), "fat".into()],
                old.data_content_fingerprint,
                Some("d".repeat(64)),
            )
            .unwrap(),
        );
        let relations: SampleRelationSet = serde_json::from_str(include_str!(
            "fixtures/multi_output_target_claims/shared/training-relations.json"
        ))
        .unwrap();
        let requirement = explain_source
            .execution_bundle
            .data_requirements
            .iter()
            .find(|requirement| requirement.key() == explain_request.data_envelope_keys[0])
            .unwrap();
        assert_eq!(
            Some(relations.fingerprint().unwrap()),
            requirement.relation_fingerprint
        );
        envelope.relation_fingerprint = requirement.relation_fingerprint.clone();
        envelope.coordinator_relations = Some(relations);
        envelope.validate().unwrap();
    }
    let explain_package = PortablePredictorPackage::from_json(include_str!(
        "fixtures/multi_output_target_claims/explain/package.json"
    ))
    .unwrap();
    preflight_loaded_predictor_replay(&explain_package, &explain_request, &explain_envelopes)
        .unwrap();
    let mut explain_controllers = RuntimeControllerRegistry::new();
    for id in explain_source
        .effective_plan
        .node_plans
        .values()
        .map(|node| node.controller_id.clone())
        .collect::<BTreeSet<_>>()
    {
        explain_controllers
            .register(Box::new(Controller {
                id,
                calls: calls.clone(),
            }))
            .unwrap();
    }
    for mutation in 0..4 {
        let mut bad = explain_envelopes.clone();
        for envelope in bad.values_mut() {
            match mutation {
                0 => wrong_set(envelope, true),
                1 => wrong_set(envelope, false),
                2 => envelope.relation_fingerprint = None,
                _ => envelope.relation_fingerprint = Some("a".repeat(64)),
            }
            envelope.validate().unwrap();
        }
        let error = execute_attached_training_replay(AttachedTrainingReplayInput {
            source: &explain_source,
            request: &explain_request,
            outcome_id: "outcome:authority:explain".into(),
            run_id: RunId::new("run:authority:explain").unwrap(),
            controllers: &explain_controllers,
            data_provider: &provider,
            artifact_store: &store,
            data_envelopes: &bad,
            warnings: Vec::new(),
            diagnostics: BTreeMap::new(),
        })
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("external_test training relations do not match owner authority"));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
    let child = PortableRefitPackageV3::from_json(CHILD).unwrap();
    let bundle = child.outcome.to_runtime_replay_bundle().unwrap();
    let (_, mut child_request, prototype) = inputs();
    let child_envelopes: BTreeMap<_, _> = bundle
        .data_requirements
        .iter()
        .map(|requirement| {
            let mut envelope = prototype.values().next().unwrap().clone();
            envelope.schema_fingerprint = requirement.schema_fingerprint.clone();
            envelope.plan_fingerprint = requirement.plan_fingerprint.clone();
            let relations: SampleRelationSet = serde_json::from_str(TARGET_RELATIONS).unwrap();
            assert_eq!(
                Some(relations.fingerprint().unwrap()),
                requirement.relation_fingerprint
            );
            envelope.relation_fingerprint = requirement.relation_fingerprint.clone();
            envelope.coordinator_relations = Some(relations);
            envelope.validate().unwrap();
            (requirement.key(), envelope)
        })
        .collect();
    child_request.source_outcome_fingerprint = child.outcome.outcome_fingerprint.clone();
    child_request.data_envelope_keys = child_envelopes.keys().cloned().collect();
    child_request.output_binding_ids = child
        .outcome
        .output_bindings
        .iter()
        .map(|binding| binding.binding_id.clone())
        .collect();
    child_request.output_binding_ids.sort();
    child_request.request_fingerprint = child_request.compute_fingerprint().unwrap();
    preflight_loaded_portable_refit_replay_v3(&child, &child_request, &child_envelopes).unwrap();
    let mut child_controllers = RuntimeControllerRegistry::new();
    for id in child
        .outcome
        .effective_plan
        .node_plans
        .values()
        .map(|node| node.controller_id.clone())
        .collect::<BTreeSet<_>>()
    {
        child_controllers
            .register(Box::new(Controller {
                id,
                calls: calls.clone(),
            }))
            .unwrap();
    }
    for mutation in 0..4 {
        let mut bad = child_envelopes.clone();
        for envelope in bad.values_mut() {
            match mutation {
                0 => wrong_set(envelope, true),
                1 => wrong_set(envelope, false),
                2 => envelope.relation_fingerprint = None,
                _ => envelope.relation_fingerprint = Some("a".repeat(64)),
            }
            envelope.validate().unwrap();
        }
        for phase in [Phase::Predict, Phase::Explain] {
            child_request.phase = phase;
            child_request.request_fingerprint = child_request.compute_fingerprint().unwrap();
            let error = execute_loaded_portable_refit_replay_v3(LoadedPortableRefitReplayInputV3 {
                package: &child,
                request: &child_request,
                outcome_id: "outcome:authority:child".into(),
                run_id: RunId::new("run:authority:child").unwrap(),
                controllers: &child_controllers,
                data_provider: &provider,
                data_envelopes: &bad,
                warnings: Vec::new(),
                diagnostics: BTreeMap::new(),
            })
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("external_test training relations do not match owner authority"));
            assert_eq!(calls.load(Ordering::SeqCst), 0);
        }
    }
}

#[test]
fn absent_b1_requirement_is_refused_by_owner_before_replay_authority() {
    let (mut package, request, envelopes) = inputs();
    package.execution_bundle.data_requirements[0].relation_fingerprint = None;
    // This is an invalid package, not a successful construction of a replayable
    // owner artifact with no requirement fingerprint. Compare the actual
    // preflight refusal with the owner's own validation, including its ordering.
    let owner_error = package.validate().unwrap_err().to_string();
    let replay_error = preflight_loaded_predictor_replay(&package, &request, &envelopes)
        .unwrap_err()
        .to_string();
    assert_eq!(replay_error, owner_error);
}

#[test]
fn editing_b1_folds_is_not_a_valid_target_missing_owner_artifact() {
    let (mut package, request, envelopes) = inputs();
    let id = envelopes
        .values()
        .next()
        .unwrap()
        .predict_cohort
        .as_ref()
        .unwrap()
        .physical_sample_ids[0]
        .clone();
    let folds = package.effective_plan.fold_set.as_mut().unwrap();
    assert!(folds.target_missing_sample_ids.is_empty());
    folds.target_missing_sample_ids.push(id);
    folds.validate().unwrap();
    // The saved plan/outcome is fingerprint-bound. A valid target-missing
    // replay needs a newly produced owner package, not this edited package.
    let owner_error = package.validate().unwrap_err().to_string();
    let replay_error = preflight_loaded_predictor_replay(&package, &request, &envelopes)
        .unwrap_err()
        .to_string();
    assert_eq!(replay_error, owner_error);
}

#[cfg(feature = "methods-optimizer")]
mod native_owner_replay {
    use super::*;
    use sha2::{Digest, Sha256};

    fn runtime() -> MethodsRuntime {
        let library = std::env::var("N4M_LIBRARY_PATH").expect("owner library path required");
        assert_eq!(std::env::var("N4M_LIB_PATH").unwrap(), library);
        let expected = std::env::var("N4M_EXPECTED_LIBRARY_SHA256")
            .expect("qualified owner library digest required");
        assert_eq!(expected.len(), 64);
        assert!(expected
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        assert_eq!(
            format!("{:x}", Sha256::digest(std::fs::read(&library).unwrap())),
            expected
        );
        MethodsRuntime::configure(&library).unwrap()
    }

    // The b1 producer projects the IO matrix's f32 cells to f64. These are
    // the same frozen identity-keyed rows; no prediction values are fabricated.
    fn dataset(input: &str, targets: Vec<String>) -> MethodsPlsDataset {
        let value: serde_json::Value = serde_json::from_str(input).unwrap();
        let source = &value["dataset"]["sources"][0];
        assert_eq!(source["name"].as_str(), Some("spectra"));
        assert_eq!(source["array"]["dtype"].as_str(), Some("float32"));
        let ids: Vec<SampleId> = source["sample_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| SampleId::new(id.as_str().unwrap()).unwrap())
            .collect();
        let rows = source["array"]["values"].as_array().unwrap();
        assert_eq!(rows.len(), ids.len());
        let cols = rows[0].as_array().unwrap().len();
        let values = rows
            .iter()
            .flat_map(|row| {
                let row = row.as_array().unwrap();
                assert_eq!(row.len(), cols);
                row.iter().map(|v| f64::from(v.as_f64().unwrap() as f32))
            })
            .collect();
        let dataset = MethodsPlsDataset {
            sample_ids: ids,
            x: MethodsPlsMatrix {
                rows: rows.len(),
                cols,
                values,
            },
            y: None,
            target_names: targets,
            y_validity_masks: None,
        };
        dataset.validate("b1 frozen input", false).unwrap();
        dataset
    }

    #[test]
    fn b1_native_replay_matches_frozen_outcome_bytes() {
        let (package, request, envelopes) = inputs();
        let data = dataset(
            include_str!("fixtures/artifact_facets_b1/inputs/calibration_dataset.json"),
            package.output_bindings[0].target_names.clone(),
        );
        let key = request.data_envelope_keys[0].clone();
        let cohort = envelopes[&key].predict_cohort.as_ref().unwrap();
        assert_eq!(data.sample_ids, cohort.physical_sample_ids);
        let methods_inputs = BTreeMap::from([(key, data)]);
        let outcome =
            execute_loaded_methods_predictor_replay(MethodsPortablePredictorReplayInput {
                package: &package,
                request: &request,
                data_envelopes: &envelopes,
                methods_inputs: &methods_inputs,
                runtime: runtime(),
                outcome_id: "outcome:b1.calibration.fixture.predict".into(),
                run_id: RunId::new("run:b1.calibration.fixture.predict").unwrap(),
                warnings: Vec::new(),
                diagnostics: BTreeMap::new(),
            })
            .unwrap();
        outcome
            .validate_against_package(&package, &request)
            .unwrap();
        assert_eq!(
            serde_json::to_vec(&outcome).unwrap().as_slice(),
            &include_bytes!("fixtures/artifact_facets_b1/outputs/calibration_replay.json")[..]
        );
    }

    #[test]
    fn inference_overlapping_training_identities_executes_native_replay() {
        let (package, request, mut envelopes) = inputs();
        let data = dataset(
            include_str!("fixtures/artifact_facets_b1/inputs/source_dataset.json"),
            package.output_bindings[0].target_names.clone(),
        );
        let key = request.data_envelope_keys[0].clone();
        let envelope = envelopes.get_mut(&key).unwrap();
        let relations = envelope.coordinator_relations.clone().unwrap();
        let folds = package.effective_plan.fold_set.as_ref().unwrap();
        assert!(data
            .sample_ids
            .iter()
            .any(|id| folds.sample_ids.contains(id)));
        let fingerprint = package.data_identities[0].data_content_fingerprint.clone();
        let cohort = PredictCohort::from_relations(
            PredictCohortRole::Inference,
            relations.clone(),
            data.target_names.clone(),
            fingerprint.clone(),
            None,
        )
        .unwrap();
        assert_eq!(data.sample_ids, cohort.physical_sample_ids);
        envelope.data_content_fingerprint = Some(fingerprint);
        envelope.target_content_fingerprint = None;
        envelope.relation_fingerprint = Some(relations.fingerprint().unwrap());
        envelope.coordinator_relations = Some(relations);
        envelope.predict_cohort = Some(cohort);
        envelope.validate().unwrap();
        preflight_loaded_predictor_replay(&package, &request, &envelopes).unwrap();
        let methods_inputs = BTreeMap::from([(key, data.clone())]);
        let outcome =
            execute_loaded_methods_predictor_replay(MethodsPortablePredictorReplayInput {
                package: &package,
                request: &request,
                data_envelopes: &envelopes,
                methods_inputs: &methods_inputs,
                runtime: runtime(),
                outcome_id: "outcome:authority:inference".into(),
                run_id: RunId::new("run:authority:inference").unwrap(),
                warnings: Vec::new(),
                diagnostics: BTreeMap::new(),
            })
            .unwrap();
        outcome
            .validate_against_package(&package, &request)
            .unwrap();
        assert_eq!(
            outcome.outputs[0].predictions[0].sample_ids,
            data.sample_ids
        );
    }
}

// These traps measure admission and refusal timing, not numerical execution.
// The CLI and C conformance controls separately exercise successful replay.
mod direct_scheduler_authority {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    const CALLBACK: &str = "authority callback reached";
    const AUTHORITY: &str = "external_test training relations do not match owner authority";

    fn fixture(
        multiple_bindings: bool,
        missing_authority: bool,
    ) -> (
        ExecutionPlan,
        ExecutionBundle,
        BTreeMap<String, ExternalDataPlanEnvelope>,
    ) {
        let mut envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        envelope.relation_fingerprint = if missing_authority {
            None
        } else {
            Some(
                envelope
                    .coordinator_relations
                    .as_ref()
                    .unwrap()
                    .fingerprint()
                    .unwrap(),
            )
        };
        let mut campaign: CampaignSpec = serde_json::from_str(include_str!(
            "../../../examples/campaign_oof_generation.json"
        ))
        .unwrap();
        let model = NodeId::new("model:base").unwrap();
        let binding = &mut campaign.data_bindings.get_mut(&model).unwrap()[0];
        binding.relation_fingerprint = envelope.relation_fingerprint.clone();
        binding.require_relations = !missing_authority;
        if multiple_bindings {
            let mut second = binding.clone();
            second.node_id = NodeId::new("transform:snv").unwrap();
            campaign
                .data_bindings
                .insert(second.node_id.clone(), vec![second]);
        }
        let manifests: Vec<ControllerManifest> =
            serde_json::from_str(include_str!("../../../examples/controller_manifests.json"))
                .unwrap();
        let mut registry = ControllerRegistry::new();
        for mut manifest in manifests {
            manifest.supported_phases.insert(Phase::Explain);
            registry.register(manifest).unwrap();
        }
        let graph =
            serde_json::from_str(include_str!("../../../examples/minimal_graph.json")).unwrap();
        let plan =
            build_execution_plan("plan:direct.authority", graph, campaign, &registry).unwrap();
        let spec: serde_json::Value = serde_json::from_str(include_str!(
            "../../../examples/fixtures/bundle/bundle_build_spec_minimal.json"
        ))
        .unwrap();
        let bundle = build_execution_bundle(
            serde_json::from_value(spec["bundle_id"].clone()).unwrap(),
            &plan,
            Some(plan.variants[0].variant_id.clone()),
            serde_json::from_value(spec["selections"].clone()).unwrap(),
            serde_json::from_value(spec["refit_artifacts"].clone()).unwrap(),
        )
        .unwrap();
        bundle.validate_against_plan(&plan).unwrap();
        envelope.schema_version = 2;
        envelope.predict_cohort = Some(cohort(PredictCohortRole::ExternalTest, "sample:heldout"));
        envelope.validate().unwrap();
        let envelopes = bundle
            .data_requirements
            .iter()
            .map(|requirement| (requirement.key(), envelope.clone()))
            .collect();
        (plan, bundle, envelopes)
    }

    fn cohort(role: PredictCohortRole, sample: &str) -> PredictCohort {
        PredictCohort::from_relations(
            role,
            SampleRelationSet {
                records: vec![SampleRelation::new(
                    ObservationId::new(format!("observation:{sample}")).unwrap(),
                    SampleId::new(sample).unwrap(),
                )],
            },
            vec!["y".into()],
            "c".repeat(64),
            (role == PredictCohortRole::ExternalTest).then(|| "d".repeat(64)),
        )
        .unwrap()
    }

    #[derive(Clone)]
    struct Trap {
        calls: Arc<AtomicUsize>,
        id: ControllerId,
    }
    impl Trap {
        fn reached<T>(&self) -> Result<T> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Err(DagMlError::RuntimeValidation(CALLBACK.into()))
        }
    }
    impl RuntimeController for Trap {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }
        fn invoke(&self, _: &NodeTask) -> Result<NodeResult> {
            self.reached()
        }
    }
    impl RuntimeArtifactStore for Trap {
        fn materialize(&self, _: &ArtifactMaterializationRequest) -> Result<HandleRef> {
            self.reached()
        }
    }
    impl RuntimeDataProvider for Trap {
        fn materialize(&self, _: &DataMaterializationRequest) -> Result<HandleRef> {
            self.reached()
        }
        fn make_view(&self, _: &DataViewRequest) -> Result<HandleRef> {
            self.reached()
        }
        fn coordinator_relations(&self, _: &DataBinding) -> Result<Option<SampleRelationSet>> {
            self.reached()
        }
        fn predict_cohort(&self, _: &DataBinding, _: Phase) -> Result<Option<PredictCohort>> {
            self.reached()
        }
    }

    fn run(
        plan: &ExecutionPlan,
        bundle: &ExecutionBundle,
        envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
        phase: Phase,
        parallel: bool,
    ) -> (DagMlError, usize) {
        let trap = Trap {
            calls: Arc::new(AtomicUsize::new(0)),
            id: ControllerId::new("controller:authority.trap").unwrap(),
        };
        let mut controllers = RuntimeControllerRegistry::new();
        for id in plan
            .node_plans
            .values()
            .map(|node| node.controller_id.clone())
            .collect::<std::collections::BTreeSet<_>>()
        {
            controllers
                .register(Box::new(Trap { id, ..trap.clone() }))
                .unwrap();
        }
        let request = ReplayPhaseRequest {
            bundle_id: bundle.bundle_id.clone(),
            phase,
            data_envelope_keys: envelopes.keys().cloned().collect(),
        };
        let replay = BundleReplayExecution {
            plan,
            bundle,
            replay_request: &request,
            prediction_cache_store: None,
            controllers: &controllers,
            data_provider: &trap,
            artifact_store: &trap,
            data_envelopes: envelopes,
        };
        let mut ctx = RunContext::new(RunId::new("run:direct.authority").unwrap(), Some(7));
        let result = if parallel {
            ParallelScheduler::new(2)
                .unwrap()
                .execute_bundle_replay(replay, &mut ctx)
        } else {
            SequentialScheduler.execute_bundle_replay(replay, &mut ctx)
        };
        (result.unwrap_err(), trap.calls.load(Ordering::SeqCst))
    }

    fn message(error: DagMlError) -> String {
        match error {
            DagMlError::RuntimeValidation(message) | DagMlError::CampaignValidation(message) => {
                message
            }
            other => panic!("unexpected refusal: {other}"),
        }
    }

    #[test]
    fn direct_replay_checks_actual_relations_before_all_callbacks_and_preserves_precedence() {
        let (plan, bundle, good) = fixture(false, false);
        let key = "model:base.x";
        for parallel in [false, true] {
            for phase in [Phase::Predict, Phase::Explain] {
                let mut bad = good.clone();
                let envelope = bad.get_mut(key).unwrap();
                wrong_set(envelope, false); // Keep the correct owner fingerprint string.
                envelope.predict_cohort = Some(cohort(PredictCohortRole::ExternalTest, "sample:1"));
                envelope.validate().unwrap();
                bundle.validate_replay_envelopes(&bad).unwrap();
                let (error, count) = run(&plan, &bundle, &bad, phase, parallel);
                assert_eq!(message(error), AUTHORITY);
                assert_eq!(count, 0);

                bad.get_mut(key).unwrap().relation_fingerprint = Some("a".repeat(64));
                let (error, count) = run(&plan, &bundle, &bad, phase, parallel);
                assert_eq!(
                    message(error),
                    "replay envelope for `model:base.x` does not match bundle data requirement"
                );
                assert_eq!(count, 0);

                // Accepted controls intentionally stop at the first host callback.
                for role in ["external_test", "inference", "v1"] {
                    let mut admitted = good.clone();
                    let envelope = admitted.get_mut(key).unwrap();
                    if role == "inference" {
                        wrong_set(envelope, false);
                        envelope.predict_cohort =
                            Some(cohort(PredictCohortRole::Inference, "sample:1"));
                    } else if role == "v1" {
                        wrong_set(envelope, false);
                        envelope.schema_version = 1;
                        envelope.predict_cohort = None;
                    }
                    envelope.validate().unwrap();
                    let (error, count) = run(&plan, &bundle, &admitted, phase, parallel);
                    assert_eq!(message(error), CALLBACK, "{role}");
                    assert_eq!(count, 1, "{role}");
                }
            }
            let mut refit = good.clone();
            wrong_set(refit.get_mut(key).unwrap(), false);
            let (error, count) = run(&plan, &bundle, &refit, Phase::Refit, parallel);
            assert_eq!(
                message(error),
                CALLBACK,
                "REFIT authority policy remains unchanged"
            );
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn missing_owner_fingerprint_is_refused_after_generic_validation() {
        let (plan, bundle, envelopes) = fixture(false, true);
        bundle.validate_replay_envelopes(&envelopes).unwrap();
        for parallel in [false, true] {
            for phase in [Phase::Predict, Phase::Explain] {
                let (error, count) = run(&plan, &bundle, &envelopes, phase, parallel);
                assert_eq!(
                    message(error),
                    "external_test requires an owner training relation fingerprint"
                );
                assert_eq!(count, 0);
            }
        }
    }

    #[test]
    fn all_requirements_are_checked_even_after_v1_or_inference() {
        let (plan, bundle, good) = fixture(true, false);
        assert_eq!(good.len(), 2);
        for first in ["external_test", "inference", "v1"] {
            let mut envelopes = good.clone();
            let envelope = envelopes.get_mut("model:base.x").unwrap();
            if first == "inference" {
                envelope.predict_cohort = Some(cohort(PredictCohortRole::Inference, "sample:1"));
            } else if first == "v1" {
                envelope.schema_version = 1;
                envelope.predict_cohort = None;
            }
            wrong_set(envelopes.get_mut("transform:snv.x").unwrap(), false);
            bundle.validate_replay_envelopes(&envelopes).unwrap();
            for parallel in [false, true] {
                for phase in [Phase::Predict, Phase::Explain] {
                    let (error, count) = run(&plan, &bundle, &envelopes, phase, parallel);
                    assert_eq!(message(error), AUTHORITY, "first requirement: {first}");
                    assert_eq!(count, 0);
                }
            }
        }
    }

    #[test]
    fn terminal_preflight_checks_all_predict_bindings_before_provider_callbacks() {
        let (plan, bundle, envelopes) = fixture(true, false);
        let mut envelope = envelopes["model:base.x"].clone();
        let selector =
            TerminalPredictionSelector::new(NodeId::new("model:base").unwrap(), "oof").unwrap();
        validate_terminal_prediction_preflight(&plan, &envelope, &selector).unwrap();
        wrong_set(&mut envelope, false);
        envelope.predict_cohort = Some(cohort(PredictCohortRole::ExternalTest, "sample:1"));
        envelope.validate().unwrap();
        assert_eq!(
            message(
                validate_terminal_prediction_preflight(&plan, &envelope, &selector).unwrap_err()
            ),
            AUTHORITY
        );
        // A selector-only check would miss authority on an upstream binding.
        let mut campaign = plan.campaign.clone();
        campaign.data_bindings.remove(&selector.node_id);
        let mut manifests = ControllerRegistry::new();
        for manifest in plan.controller_manifests.values() {
            manifests.register(manifest.clone()).unwrap();
        }
        let upstream_only = build_execution_plan(
            "plan:terminal.upstream.authority",
            plan.graph_plan.graph.clone(),
            campaign,
            &manifests,
        )
        .unwrap();
        assert!(upstream_only.node_plans[&selector.node_id]
            .data_bindings
            .is_empty());
        validate_terminal_prediction_preflight(
            &upstream_only,
            &envelopes["model:base.x"],
            &selector,
        )
        .unwrap();
        assert_eq!(
            message(
                validate_terminal_prediction_preflight(&upstream_only, &envelope, &selector)
                    .unwrap_err()
            ),
            AUTHORITY
        );
        let trap = Trap {
            calls: Arc::new(AtomicUsize::new(0)),
            id: ControllerId::new("controller:authority.trap").unwrap(),
        };
        let controllers = RuntimeControllerRegistry::new();
        let mut ctx = RunContext::new(RunId::new("run:terminal.authority").unwrap(), Some(7));
        let error = execute_terminal_prediction(
            TerminalPredictionReplay {
                plan: &plan,
                bundle: &bundle,
                envelope: &envelope,
                selector: &selector,
                controllers: &controllers,
                data_provider: &trap,
                artifact_store: &trap,
            },
            &mut ctx,
        )
        .unwrap_err();
        assert_eq!(message(error), AUTHORITY);
        assert_eq!(trap.calls.load(Ordering::SeqCst), 0);

        let invalid = TerminalPredictionSelector::new(selector.node_id.clone(), "missing").unwrap();
        assert_eq!(
            message(
                validate_terminal_prediction_preflight(&plan, &envelope, &invalid).unwrap_err()
            ),
            "terminal PREDICT selector `model:base` has no output port `missing`"
        );
    }
}
