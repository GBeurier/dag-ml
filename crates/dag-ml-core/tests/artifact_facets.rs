use std::collections::{BTreeMap, BTreeSet};

use dag_ml_core::*;
use serde_json::json;
use sha2::Digest;

const OUTCOME: &[u8] = include_bytes!("fixtures/artifact_facets/training_outcome.json");
const INITIAL: &[u8] = include_bytes!("fixtures/artifact_facets/initial_refit.json");

fn source() -> (TrainingOutcome, PortablePredictorPackage) {
    let outcome = TrainingOutcome::from_json(std::str::from_utf8(OUTCOME).unwrap()).unwrap();
    let package = outcome
        .to_portable_predictor_package(
            "package:facets",
            FittedArtifactMode::AllowHostSidecar,
            ArtifactLoadMode::HostSidecar,
        )
        .unwrap();
    (outcome, package)
}

fn export() -> ArtifactFacetExport {
    let (outcome, package) = source();
    build_training_artifact_facets(&outcome, &package).unwrap()
}

fn proof(id: &str, version: u32, bytes: &[u8]) -> ValidatedOwnerArtifact {
    validate_owner_facet_bytes(&FacetContentType::new(id, version), bytes)
        .unwrap()
        .unwrap()
}

fn replace_proof(export: &mut ArtifactFacetExport, id: &str, proof: ValidatedOwnerArtifact) {
    let artifact = export
        .graph
        .artifacts
        .iter_mut()
        .find(|artifact| artifact.id == id)
        .unwrap();
    artifact.identity = proof.identity().clone();
    artifact.content_type = proof.content_type().clone();
    if let ArtifactFacetStorage::Embedded { root_member, .. } = &artifact.storage {
        export
            .members
            .insert(root_member.clone(), proof.original_bytes().to_vec());
    } else {
        artifact.storage = ArtifactFacetStorage::Reference {
            root_sha256: Some(proof.raw_sha256().into()),
        };
    }
    export.context.owner_proofs.insert(id.into(), proof);
}

#[test]
fn direct_training_builder_preserves_current_owner_closure_and_honest_evaluation_status() {
    let export = export();
    let report = export.validate().unwrap();
    assert!(report.len() >= 10);
    assert!(
        report
            .iter()
            .filter(|link| link.verdict.status == FacetSemanticStatus::Validated)
            .count()
            >= 9
    );
    let evaluation = export
        .graph
        .links
        .iter()
        .find(|link| link.relation == ArtifactFacetRelation::Evaluates)
        .unwrap();
    assert_eq!(
        report
            .iter()
            .find(|link| link.link_id == evaluation.id)
            .unwrap()
            .verdict
            .status,
        FacetSemanticStatus::Unverified
    );
    let recipe = &export.context.owner_proofs["recipe"];
    assert_eq!(
        recipe.identity().fingerprint_profile,
        FACET_GRAPH_SPEC_PROFILE
    );
    assert_ne!(recipe.identity().fingerprint, recipe.raw_sha256());
    let parsed =
        ArtifactFacetGraph::from_json(&serde_json::to_string(&export.graph).unwrap()).unwrap();
    assert_eq!(parsed, export.graph);
    require_facet_operation(
        "score-inspect",
        "score",
        &[],
        &export.graph,
        &export.context,
    )
    .unwrap();
    for operation in ["predict", "compile", "refit", "unknown"] {
        assert!(
            require_facet_operation(operation, "model", &[], &export.graph, &export.context)
                .is_err()
        );
    }
}

#[test]
fn byte_builder_keeps_original_whitespace_and_token_spelling() {
    let (_, package) = source();
    let package = serde_json::to_vec_pretty(&package).unwrap();
    let mut original = b" \n\t".to_vec();
    original.extend_from_slice(OUTCOME);
    original.extend_from_slice(b"\n ");
    let export = build_training_artifact_facets_from_bytes(&original, &package).unwrap();
    assert_eq!(export.members["dagml/result.json"], original);
    assert_eq!(export.members["dagml/model.json"], package);
    assert_eq!(
        export.context.owner_proofs["evidence"].original_bytes(),
        original
    );
}

#[test]
fn strict_bytes_preserve_u64_and_integer_vs_binary64_and_reject_ambiguity() {
    let (outcome, _) = source();
    let mut graph = serde_json::to_value(&outcome.effective_plan.graph_plan.graph).unwrap();
    graph["metadata"]["large"] = json!(u64::MAX);
    let integer = serde_json::to_vec(&graph).unwrap();
    let proof_integer = proof(FACET_GRAPH_SPEC, 1, &integer);
    assert_eq!(proof_integer.original_bytes(), integer);
    assert!(std::str::from_utf8(proof_integer.original_bytes())
        .unwrap()
        .contains("18446744073709551615"));
    graph["metadata"]["large"] = json!(1.0);
    let float = proof(FACET_GRAPH_SPEC, 1, &serde_json::to_vec(&graph).unwrap());
    graph["metadata"]["large"] = json!(1);
    let int = proof(FACET_GRAPH_SPEC, 1, &serde_json::to_vec(&graph).unwrap());
    assert_ne!(float.identity().fingerprint, int.identity().fingerprint);
    let base = serde_json::to_string(&graph).unwrap();
    for bad in [
        base.replacen('{', "{\"id\":\"duplicate\",", 1),
        base.replacen('{', "{\"surprise\":1,", 1),
        base.replacen('{', "{\"é\":1,\"e\\u0301\":2,", 1),
        base.replace("\"large\":1", "\"large\":18446744073709551616"),
    ] {
        assert!(
            validate_owner_facet_bytes(&FacetContentType::new(FACET_GRAPH_SPEC, 1), bad.as_bytes())
                .is_err(),
            "accepted {bad}"
        );
    }
    assert!(
        validate_owner_facet_bytes(&FacetContentType::new(FACET_GRAPH_SPEC, 1), b"\xff").is_err()
    );
    let positional = json!(["graph", {}, [], [], null, {}]);
    assert!(validate_owner_facet_bytes(
        &FacetContentType::new(FACET_GRAPH_SPEC, 1),
        &serde_json::to_vec(&positional).unwrap()
    )
    .is_err());
    assert!(
        validate_owner_facet_bytes(&FacetContentType::new(FACET_GRAPH_SPEC, 2), &integer).is_err()
    );
}

#[test]
fn profiles_use_content_type_and_payload_not_raw_sha_or_invented_self_fields() {
    for profile in [
        FACET_GRAPH_SPEC_PROFILE,
        FACET_EXECUTION_BUNDLE_PROFILE,
        FACET_SCORE_SET_PROFILE,
        FACET_PREDICTION_CACHES_PROFILE,
        FACET_EXECUTION_PLAN_PROFILE,
    ] {
        assert!(DAG_FACET_FINGERPRINT_PROFILES.contains(&profile));
    }
    let content1 = FacetContentType::new(FACET_SCORE_SET, 1);
    let bytes1 = br#"{"schema_version":1,"plan_id":"plan:one","reports":[]}"#;
    let a = validate_owner_facet_bytes(&content1, bytes1)
        .unwrap()
        .unwrap();
    let bytes2 = br#"{"schema_version":2,"plan_id":"plan:one","reports":[]}"#;
    let b = proof(FACET_SCORE_SET, 2, bytes2);
    assert_ne!(a.identity().fingerprint, b.identity().fingerprint);
    // Independent construction of the documented TCV1 domain.
    let domain = format!(
        "{{\"profile\":\"{}\",\"content_type\":{{\"id\":\"{}\",\"version\":1}},\"payload\":{}}}",
        FACET_SCORE_SET_PROFILE,
        FACET_SCORE_SET,
        std::str::from_utf8(bytes1).unwrap()
    );
    assert_eq!(
        a.identity().fingerprint,
        parse_typed_json(&domain).unwrap().fingerprint().unwrap()
    );
    assert_ne!(
        a.identity().fingerprint,
        parse_typed_json_bytes(bytes1)
            .unwrap()
            .fingerprint()
            .unwrap()
    );
    let wrong_domain = domain.replace(FACET_SCORE_SET, FACET_EXECUTION_BUNDLE);
    assert_ne!(
        a.identity().fingerprint,
        parse_typed_json(&wrong_domain)
            .unwrap()
            .fingerprint()
            .unwrap()
    );
    let reordered = br#"{ "reports":[], "plan_id":"plan:one", "schema_version":1 }"#;
    let same = proof(FACET_SCORE_SET, 1, reordered);
    assert_eq!(a.identity(), same.identity());
    assert_ne!(a.raw_sha256(), same.raw_sha256());
    assert!(validate_owner_facet_bytes(
        &FacetContentType::new(FACET_SCORE_SET, 1),
        br#"{"schema_version":1,"plan_id":"p","reports":[],"score_fingerprint":"fake"}"#
    )
    .is_err());
}

#[test]
fn embedded_and_reference_require_every_identity_field() {
    let export = export();
    let wire = serde_json::to_value(&export.graph).unwrap();
    for artifact_index in [
        0,
        export
            .graph
            .artifacts
            .iter()
            .position(|a| a.id == "evidence")
            .unwrap(),
    ] {
        for field in [
            "owner",
            "schema",
            "schema_version",
            "fingerprint_profile",
            "fingerprint",
        ] {
            let mut bad = wire.clone();
            bad["artifacts"][artifact_index]["identity"]
                .as_object_mut()
                .unwrap()
                .remove(field);
            assert!(
                ArtifactFacetGraph::from_json(&bad.to_string()).is_err(),
                "missing {field}"
            );
        }
        for field in ["owner", "schema", "fingerprint_profile", "fingerprint"] {
            let mut bad = wire.clone();
            bad["artifacts"][artifact_index]["identity"][field] = json!("");
            assert!(ArtifactFacetGraph::from_json(&bad.to_string()).is_err());
        }
        let mut bad = wire.clone();
        bad["artifacts"][artifact_index]["identity"]["schema_version"] = json!(0);
        assert!(ArtifactFacetGraph::from_json(&bad.to_string()).is_err());
    }
    let mut bad = wire.clone();
    bad["artifacts"][0]
        .as_object_mut()
        .unwrap()
        .remove("requirements");
    assert!(ArtifactFacetGraph::from_json(&bad.to_string()).is_err());
    let mut bad = wire;
    bad["artifacts"][0]["kind"] = json!("calibration");
    assert!(ArtifactFacetGraph::from_json(&bad.to_string()).is_err());
}

#[test]
fn relation_vocabulary_directions_and_closure_are_closed() {
    let export = export();
    let mut graph = export.graph.clone();
    let link = &mut graph.links[0];
    std::mem::swap(&mut link.source_artifact, &mut link.target_artifact);
    assert!(graph.validate_structure().is_err());
    let mut wire = serde_json::to_value(&export.graph).unwrap();
    wire["links"][0]["relation"] = json!("calibrates");
    assert!(ArtifactFacetGraph::from_json(&wire.to_string()).is_err());
    let mut graph = export.graph.clone();
    let mut duplicate = graph.links[0].clone();
    duplicate.id = "different-label".into();
    graph.links.push(duplicate);
    assert!(graph.validate_structure().is_err());
    let mut graph = export.graph.clone();
    graph.artifacts.push(graph.artifacts[0].clone());
    assert!(graph.validate_structure().is_err());
    let mut graph = export.graph.clone();
    graph.links[0].target_artifact = "undeclared".into();
    assert!(graph.validate_structure().is_err());
    let mut graph = export.graph.clone();
    graph.links[0].target_artifact = graph.links[0].source_artifact.clone();
    assert!(graph.validate_structure().is_err());
    let mut graph = export.graph.clone();
    graph.artifacts[0].requirements = Some(FacetRequirementReference {
        artifact_id: "absent".into(),
    });
    assert!(graph.validate_structure().is_err());
    graph.artifacts[0]
        .requirements
        .as_mut()
        .unwrap()
        .artifact_id = "score".into();
    graph.validate_structure().unwrap();
    // Duplicate physical ownership is forbidden independently of equal bytes.
    graph.artifacts[1].storage = graph.artifacts[0].storage.clone();
    assert!(graph.validate_structure().is_err());
}

#[test]
fn current_owner_proof_is_required_and_declared_identity_mutations_refuse() {
    for id in ["recipe", "model", "result", "score", "evidence", "run"] {
        let mut export = export();
        let artifact = export
            .graph
            .artifacts
            .iter_mut()
            .find(|a| a.id == id)
            .unwrap();
        artifact.identity.fingerprint = "0".repeat(64);
        assert!(
            validate_artifact_facet_links(&export.graph, &export.context).is_err(),
            "mutated {id}"
        );
    }
    let mut export = export();
    let artifact = export
        .graph
        .artifacts
        .iter_mut()
        .find(|a| a.id == "evidence")
        .unwrap();
    artifact.storage = ArtifactFacetStorage::Reference {
        root_sha256: Some("0".repeat(64)),
    };
    assert!(validate_artifact_facet_links(&export.graph, &export.context).is_err());
}

#[test]
fn valid_but_different_owner_payloads_produce_mismatch_and_refuse_action() {
    let mut export = export();
    let (outcome, _) = source();
    let mut graph = outcome.effective_plan.graph_plan.graph.clone();
    graph.id.push_str("-different");
    replace_proof(
        &mut export,
        "recipe",
        proof(FACET_GRAPH_SPEC, 1, &serde_json::to_vec(&graph).unwrap()),
    );
    assert!(inspect_artifact_facet_links(&export.graph, &export.context)
        .unwrap()
        .iter()
        .any(|l| l.verdict.status == FacetSemanticStatus::Mismatch));
    assert!(export.validate().is_err());
    assert!(
        require_facet_operation("inspect", "model", &[], &export.graph, &export.context).is_err()
    );

    let mut export = self::export();
    let mut scores = outcome.score_set.clone();
    scores.plan_id.push_str("-different");
    replace_proof(
        &mut export,
        "score",
        proof(
            FACET_SCORE_SET,
            scores.schema_version,
            &serde_json::to_vec(&scores).unwrap(),
        ),
    );
    assert!(export.validate().is_err());

    let (mut changed, _) = source();
    changed.outcome_id.push_str("-different");
    changed.outcome_fingerprint = changed.compute_fingerprint().unwrap();
    changed.validate().unwrap();
    let (_, package) = source();
    assert!(build_training_artifact_facets(&changed, &package).is_err());
    let mut package = package;
    package.package_fingerprint = "0".repeat(64);
    assert!(build_training_artifact_facets(&outcome, &package).is_err());
}

fn reference(id: &str, kind: ArtifactFacetKind) -> ArtifactFacet {
    ArtifactFacet {
        id: id.into(),
        kind,
        owner: "external-owner".into(),
        content_type: FacetContentType::new("owner.contract", 1),
        identity: OwnerIdentity {
            owner: "external-owner".into(),
            schema: "owner.contract".into(),
            schema_version: 1,
            fingerprint_profile: "owner-semantic-v1".into(),
            fingerprint: "semantic-content".into(),
        },
        storage: ArtifactFacetStorage::Reference { root_sha256: None },
        requirements: None,
    }
}

#[test]
fn partial_result_and_score_exports_accept_complete_unresolved_references() {
    for root in ["result", "score"] {
        let mut export = export();
        for artifact in &mut export.graph.artifacts {
            if artifact.id != root {
                artifact.storage = ArtifactFacetStorage::Reference { root_sha256: None };
            }
        }
        export.context.owner_proofs.retain(|id, _| id == root);
        let report = validate_artifact_facet_links(&export.graph, &export.context).unwrap();
        assert!(report
            .iter()
            .all(|link| link.verdict.status == FacetSemanticStatus::Unverified));
        require_facet_operation("inspect", root, &[], &export.graph, &export.context).unwrap();
        assert!(
            require_facet_operation("inspect", "recipe", &[], &export.graph, &export.context)
                .is_err()
        );
    }
}

#[test]
fn multiple_dataset_cohorts_and_residuals_stay_unverified_without_real_owner_evidence() {
    let mut export = export();
    for id in ["train-dataset", "predict-dataset"] {
        export
            .graph
            .artifacts
            .push(reference(id, ArtifactFacetKind::Dataset));
        export.graph.links.push(ArtifactFacetLink {
            id: id.into(),
            owner: "dag-ml".into(),
            relation: ArtifactFacetRelation::DatasetOf,
            source_artifact: id.into(),
            target_artifact: "run".into(),
        });
    }
    export
        .graph
        .artifacts
        .push(reference("residual", ArtifactFacetKind::Residual));
    export.graph.artifacts.push(reference(
        "external-prediction",
        ArtifactFacetKind::Prediction,
    ));
    export.graph.links.push(ArtifactFacetLink {
        id: "residual-link".into(),
        owner: "dag-ml".into(),
        relation: ArtifactFacetRelation::ResidualOf,
        source_artifact: "residual".into(),
        target_artifact: "external-prediction".into(),
    });
    for link in validate_artifact_facet_links(&export.graph, &export.context)
        .unwrap()
        .iter()
        .filter(|link| {
            ["train-dataset", "predict-dataset", "residual-link"].contains(&link.link_id.as_str())
        })
    {
        assert_eq!(link.verdict.status, FacetSemanticStatus::Unverified);
    }
    assert!(validate_owner_facet_bytes(
        &FacetContentType::new("nirs4all-methods.n4mm", 3),
        b"opaque-not-decoded"
    )
    .unwrap()
    .is_none());
}

struct RejectDataset;
impl FacetDatasetBindingValidator for RejectDataset {
    fn validate_dataset_binding(
        &self,
        _: &ArtifactFacet,
        _: Option<&[u8]>,
        _: &ValidatedOwnerArtifact,
    ) -> FacetResult<FacetLinkVerdict> {
        Ok(FacetLinkVerdict {
            status: FacetSemanticStatus::Mismatch,
            reason: "actual IO-to-binding composition disagrees".into(),
        })
    }
}

#[test]
fn supplied_composition_mismatch_cannot_be_downgraded_to_unverified() {
    let mut export = export();
    export
        .graph
        .artifacts
        .push(reference("dataset", ArtifactFacetKind::Dataset));
    export.graph.links.push(ArtifactFacetLink {
        id: "dataset".into(),
        owner: "dag-ml".into(),
        relation: ArtifactFacetRelation::DatasetOf,
        source_artifact: "dataset".into(),
        target_artifact: "model".into(),
    });
    let validator = RejectDataset;
    let context = export.context.with_dataset_binding_validator(&validator);
    assert!(inspect_artifact_facet_links(&export.graph, &context)
        .unwrap()
        .iter()
        .any(|l| l.verdict.status == FacetSemanticStatus::Mismatch));
    assert!(validate_artifact_facet_links(&export.graph, &context).is_err());
}

#[test]
fn host_trust_and_runtime_capabilities_are_checked_without_any_loader() {
    let export = export();
    let proof = &export.context.owner_proofs["model"];
    let capabilities = BTreeSet::new();
    let refusal = check_facet_hydration_requirements(proof, &capabilities, false).unwrap_err();
    assert!(matches!(refusal, FacetRefusal::MissingTrust));
    assert!(check_facet_hydration_requirements(proof, &capabilities, true).is_err());
    let (_, package) = source();
    let mut capabilities: BTreeSet<String> = package
        .execution_bundle
        .refit_artifacts
        .iter()
        .flat_map(|record| {
            [
                format!("controller:{}", record.controller_id),
                format!("serializer:{:?}", record.artifact.backend.unwrap()).to_lowercase(),
            ]
        })
        .collect();
    for record in &package.execution_bundle.refit_artifacts {
        if let (Some(plugin), Some(version)) =
            (&record.artifact.plugin, &record.artifact.plugin_version)
        {
            capabilities.insert(format!("plugin:{plugin}:{version}"));
        }
    }
    check_facet_hydration_requirements(proof, &capabilities, true).unwrap();
    let states = inspect_facet_model_states(proof).unwrap();
    assert!(!states.is_empty());
    assert!(states.iter().all(
        |state| state.verdict.status == FacetSemanticStatus::Unverified
            && state.native_descriptor.is_none()
    ));
    // These checks have no callback, resolver or fitted-object handle to invoke.
    assert!(
        require_facet_operation("predict", "model", &[], &export.graph, &export.context).is_err()
    );
}

#[test]
fn initial_refit_builder_has_no_parent_or_scores_and_rejects_mutated_identity() {
    let package =
        InitialFullRefitPackage::from_json(std::str::from_utf8(INITIAL).unwrap()).unwrap();
    let export = build_initial_refit_artifact_facets(&package).unwrap();
    export.validate().unwrap();
    assert!(export.graph.artifacts.iter().all(|a| !matches!(
        a.kind,
        ArtifactFacetKind::Score | ArtifactFacetKind::TrainingEvidence
    )));
    assert!(export
        .graph
        .links
        .iter()
        .all(|l| l.relation != ArtifactFacetRelation::RefitOf
            && l.relation != ArtifactFacetRelation::Evaluates));
    let mut bad = package;
    bad.package_fingerprint = "0".repeat(64);
    assert!(build_initial_refit_artifact_facets(&bad).is_err());
}

#[test]
fn exact_member_map_refuses_mutated_or_extra_physical_bytes() {
    let mut export = export();
    export
        .members
        .get_mut("dagml/result.json")
        .unwrap()
        .push(b' ');
    assert!(export.validate().is_err());
    let mut export = self::export();
    export.members.insert("unowned.bin".into(), vec![1]);
    assert!(export.validate().is_err());
}

#[test]
fn owner_self_fingerprints_reject_raw_numeric_token_mutation() {
    let (mut outcome, _) = source();
    outcome.diagnostics.insert("integer-token".into(), json!(1));
    outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
    let bytes = serde_json::to_string(&outcome).unwrap();
    proof(
        FACET_TRAINING_OUTCOME,
        outcome.schema_version,
        bytes.as_bytes(),
    );
    let mutated = bytes.replace("\"integer-token\":1", "\"integer-token\":1.0");
    assert_ne!(bytes, mutated);
    assert!(validate_owner_facet_bytes(
        &FacetContentType::new(FACET_TRAINING_OUTCOME, outcome.schema_version),
        mutated.as_bytes()
    )
    .is_err());
}

#[test]
fn immutable_owner_capture_validates_without_original_process_or_container() {
    const FACETS: &str = include_str!("fixtures/artifact_facets/capture/facets.json");
    const RESULT: &[u8] = include_bytes!("fixtures/artifact_facets/capture/dagml/result.json");
    const MODEL: &[u8] = include_bytes!("fixtures/artifact_facets/capture/dagml/model.json");
    const INVENTORY: &str = include_str!("fixtures/artifact_facets/capture/inventory.json");
    let graph = ArtifactFacetGraph::from_json(FACETS).unwrap();
    let mut context = FacetValidationContext::default();
    for (id, content, bytes) in [
        ("result", FACET_TRAINING_OUTCOME, RESULT),
        ("model", FACET_PREDICTOR_PACKAGE, MODEL),
    ] {
        context
            .owner_proofs
            .insert(id.into(), proof(content, 2, bytes));
    }
    validate_artifact_facet_links(&graph, &context).unwrap();
    let inventory: BTreeMap<String, serde_json::Value> = serde_json::from_str(INVENTORY).unwrap();
    for (path, bytes) in [("dagml/result.json", RESULT), ("dagml/model.json", MODEL)] {
        assert_eq!(inventory[path]["byte_length"], bytes.len());
        assert_eq!(
            inventory[path]["sha256"],
            format!("{:x}", sha2::Sha256::digest(bytes))
        );
    }
    let fresh = build_training_artifact_facets_from_bytes(RESULT, MODEL).unwrap();
    fresh.validate().unwrap();
    assert_eq!(fresh.graph, graph);
}

#[cfg(feature = "methods-optimizer")]
#[test]
fn actual_methods_inspector_reports_n4mm3_without_generic_dag_decoding() {
    let library = std::env::var("N4M_LIBRARY_PATH")
        .expect("optional native test requires an explicit actual Methods library");
    assert_eq!(std::env::var("N4M_LIB_PATH").unwrap(), library);
    let runtime_bytes = std::fs::read(&library).unwrap();
    assert_eq!(
        format!("{:x}", sha2::Sha256::digest(&runtime_bytes)),
        std::env::var("N4M_EXPECTED_LIBRARY_SHA256")
            .expect("native test requires the qualified owner library SHA256")
    );
    let _runtime = MethodsRuntime::configure(&library).unwrap();
    let context = n4m::Context::new().unwrap();
    let mut config = n4m::Config::new().unwrap();
    config.set_n_components(1).unwrap();
    config.set_snv_savgol_pipeline(3, 2).unwrap();
    let x = [
        1.0, 1.5, 2.0, 2.5, 2.0, 3.0, 4.0, 5.0, 3.0, 5.0, 8.0, 13.0, 4.0, 7.0, 11.0, 16.0,
    ];
    let y = [1.0, 2.0, 3.0, 4.0];
    let state = n4m::Model::fit(
        &context,
        &config,
        n4m::MatrixRef::row_major(&x, 4, 4).unwrap(),
        n4m::MatrixRef::row_major(&y, 4, 1).unwrap(),
    )
    .unwrap()
    .export_n4mm()
    .unwrap();
    let owner = ControllerId::new(hpo::METHODS_PLS_CONTROLLER_ID).unwrap();
    let descriptor = inspect_methods_native_predictor_descriptor_v1(&owner, &state).unwrap();
    assert_eq!(descriptor.format_version, 3);
    assert_eq!(descriptor.dimensions.n_features, 4);
    assert_eq!(descriptor.dimensions.n_targets, 1);
    assert!(descriptor.pipeline.is_some());
    // These remain owner-runtime facts. Generic DAG never parses Methods bytes.
    assert!(
        validate_owner_facet_bytes(&FacetContentType::new("nirs4all-methods.n4mm", 3), &state)
            .unwrap()
            .is_none()
    );
    let mut corrupt = state;
    corrupt[0] ^= 0xff;
    assert!(inspect_methods_native_predictor_descriptor_v1(&owner, &corrupt).is_err());
}

#[test]
fn execution_bundle_without_owner_fingerprinted_run_identity_cannot_validate_producer_run() {
    let mut export = export();
    let bundle = export.context.owner_proofs["bundle"].clone();
    replace_proof(&mut export, "run", bundle);
    let report = validate_artifact_facet_links(&export.graph, &export.context).unwrap();
    for link in export
        .graph
        .links
        .iter()
        .filter(|link| link.relation == ArtifactFacetRelation::ProducedBy)
    {
        assert_eq!(
            report
                .iter()
                .find(|result| result.link_id == link.id)
                .unwrap()
                .verdict
                .status,
            FacetSemanticStatus::Unverified
        );
    }
}

#[test]
fn cache_identity_normalizes_owner_accepted_float_tokens() {
    let mut export = export();
    let original = &export.context.owner_proofs["prediction"];
    let text = std::str::from_utf8(original.original_bytes()).unwrap();
    let changed_text = text.replacen("50420.0", "50420", 1);
    assert_ne!(text, changed_text);
    let owner: BundlePredictionCachePayloadSet = serde_json::from_str(&changed_text).unwrap();
    owner.validate().unwrap();
    let changed = proof(FACET_PREDICTION_CACHES, 2, changed_text.as_bytes());
    assert_eq!(original.prediction_caches(), changed.prediction_caches());
    assert_eq!(original.identity(), changed.identity());
    assert_ne!(original.raw_sha256(), changed.raw_sha256());
    assert_eq!(changed.original_bytes(), changed_text.as_bytes());
    assert_eq!(
        changed.identity().fingerprint_profile,
        FACET_PREDICTION_CACHES_PROFILE
    );
    replace_proof(&mut export, "prediction", changed);
    validate_artifact_facet_links(&export.graph, &export.context).unwrap();
}

#[test]
fn requirement_wire_is_required_nullable_and_minimal_in_public_serde() {
    let graph = export().graph;
    let wire = serde_json::to_value(&graph).unwrap();
    let mut missing = wire.clone();
    missing["artifacts"][0]
        .as_object_mut()
        .unwrap()
        .remove("requirements");
    assert!(serde_json::from_value::<ArtifactFacetGraph>(missing.clone()).is_err());
    assert!(ArtifactFacetGraph::from_json(&missing.to_string()).is_err());
    serde_json::from_value::<ArtifactFacetGraph>(wire.clone()).unwrap();
    for invalid in [
        json!(["score"]),
        json!({}),
        json!({"artifact_id":"score","content_type":{"id":"duplicate","version":1}}),
        json!({"artifact_id":"score","extra":true}),
    ] {
        let mut bad = wire.clone();
        bad["artifacts"][0]["requirements"] = invalid;
        assert!(serde_json::from_value::<ArtifactFacetGraph>(bad.clone()).is_err());
        assert!(ArtifactFacetGraph::from_json(&bad.to_string()).is_err());
    }
    let mut valid = wire.clone();
    valid["artifacts"][0]["requirements"] = json!({"artifact_id":"score"});
    let parsed = ArtifactFacetGraph::from_json(&valid.to_string()).unwrap();
    assert_eq!(serde_json::to_value(&parsed).unwrap(), valid);
    valid["artifacts"][0]["requirements"] = json!({"artifact_id":"undeclared"});
    assert!(ArtifactFacetGraph::from_json(&valid.to_string()).is_err());
    // Structural identity schema/version are independent of content_type.
    let mut independent = wire;
    independent["artifacts"][0]["identity"]["schema"] = json!("owner.different_identity");
    independent["artifacts"][0]["identity"]["schema_version"] = json!(7);
    ArtifactFacetGraph::from_json(&independent.to_string()).unwrap();
}

#[test]
fn reference_root_digest_is_optional_but_never_null_or_malformed() {
    for invalid in [
        json!(null),
        json!(""),
        json!("F".repeat(64)),
        json!("z".repeat(64)),
        json!(123),
    ] {
        let bad = json!({"kind":"reference","root_sha256":invalid});
        assert!(serde_json::from_value::<ArtifactFacetStorage>(bad).is_err());
    }
    for valid in [
        json!({"kind":"reference"}),
        json!({"kind":"reference","root_sha256":"a".repeat(64)}),
    ] {
        let parsed: ArtifactFacetStorage = serde_json::from_value(valid.clone()).unwrap();
        assert_eq!(serde_json::to_value(parsed).unwrap(), valid);
    }
    let mut graph = serde_json::to_value(export().graph).unwrap();
    graph["artifacts"][0]["storage"] = json!({"kind":"reference","root_sha256":null});
    assert!(ArtifactFacetGraph::from_json(&graph.to_string()).is_err());
}

#[test]
fn new_profiles_normalize_valid_defaults_and_nulls_without_replacing_bytes() {
    let mut export = export();
    for id in ["recipe", "score", "bundle"] {
        let original = &export.context.owner_proofs[id];
        let content = original.content_type().clone();
        let mut alternate: serde_json::Value =
            serde_json::from_slice(original.original_bytes()).unwrap();
        match id {
            "recipe" => {
                let port = &mut alternate["interface"]["inputs"][0];
                assert_eq!(port["description"], "");
                port.as_object_mut().unwrap().remove("description");
                port["unit_level"] = json!(null);
                GraphSpec::from_json(&alternate.to_string()).unwrap();
            }
            "score" => {
                for report in alternate["reports"].as_array_mut().unwrap() {
                    report["target_domain"] = json!(null);
                    report["grouping_key"] = json!(null);
                    report["variant_label"] = json!(null);
                }
                serde_json::from_value::<ScoreSet>(alternate.clone())
                    .unwrap()
                    .validate()
                    .unwrap();
            }
            "bundle" => {
                assert_eq!(alternate["unsafe_flags"], json!([]));
                alternate.as_object_mut().unwrap().remove("unsafe_flags");
                alternate["conformal_calibration"] = json!(null);
                ExecutionBundle::from_json(&alternate.to_string()).unwrap();
            }
            _ => unreachable!(),
        }
        let bytes = format!(
            " \n{}\n\t",
            serde_json::to_string_pretty(&alternate).unwrap()
        )
        .into_bytes();
        let normalized = validate_owner_facet_bytes(&content, &bytes)
            .unwrap()
            .unwrap();
        assert_eq!(original.identity(), normalized.identity(), "{id}");
        assert_ne!(original.raw_sha256(), normalized.raw_sha256(), "{id}");
        assert_eq!(normalized.original_bytes(), bytes);
        replace_proof(&mut export, id, normalized);
        export.validate().unwrap();
    }
}

#[test]
fn strict_original_preflight_rejects_nested_keys_and_out_of_range_numbers() {
    let (outcome, _) = source();
    let mut graph = outcome.effective_plan.graph_plan.graph.clone();
    graph
        .metadata
        .insert("signed-bound".into(), json!(i64::MIN));
    graph
        .metadata
        .insert("unsigned-bound".into(), json!(u64::MAX));
    let text = serde_json::to_string(&graph).unwrap();
    let retained = proof(FACET_GRAPH_SPEC, 1, text.as_bytes());
    assert!(std::str::from_utf8(retained.original_bytes())
        .unwrap()
        .contains("18446744073709551615"));
    for bad in [
        text.replace(
            "\"signed-bound\":-9223372036854775808",
            "\"signed-bound\":0,\"signed-bound\":1",
        ),
        text.replace("-9223372036854775808", "-9223372036854775809"),
        text.replace("18446744073709551615", "18446744073709551616"),
        text.replace("18446744073709551615", "1e400"),
    ] {
        assert_ne!(bad, text);
        assert!(validate_owner_facet_bytes(
            &FacetContentType::new(FACET_GRAPH_SPEC, 1),
            bad.as_bytes()
        )
        .is_err());
    }
}

#[test]
fn known_payload_kind_relabeling_refuses_before_scientific_dispatch() {
    for (id, kind) in [
        ("score", ArtifactFacetKind::Model),
        ("recipe", ArtifactFacetKind::Score),
        ("result", ArtifactFacetKind::Model),
        ("model", ArtifactFacetKind::TrainingEvidence),
        ("prediction", ArtifactFacetKind::Residual),
        ("plan", ArtifactFacetKind::Recipe),
    ] {
        let mut export = export();
        export.graph.links.clear();
        export
            .graph
            .artifacts
            .iter_mut()
            .find(|a| a.id == id)
            .unwrap()
            .kind = kind;
        let refusal = validate_artifact_facet_links(&export.graph, &export.context).unwrap_err();
        assert!(
            matches!(refusal, FacetRefusal::Mismatch { .. }),
            "{id}: {refusal}"
        );
    }
    for storage_kind in [ArtifactFacetKind::Attachment, ArtifactFacetKind::Provenance] {
        let mut export = export();
        export.graph.links.clear();
        export
            .graph
            .artifacts
            .iter_mut()
            .find(|a| a.id == "score")
            .unwrap()
            .kind = storage_kind;
        validate_artifact_facet_links(&export.graph, &export.context).unwrap();
        require_facet_operation("inspect", "score", &[], &export.graph, &export.context).unwrap();
        assert!(require_facet_operation(
            "score-inspect",
            "score",
            &[],
            &export.graph,
            &export.context
        )
        .is_err());
        export.graph.links.push(ArtifactFacetLink {
            id: "storage-score".into(),
            owner: "dag-ml".into(),
            relation: ArtifactFacetRelation::ProducedBy,
            source_artifact: "score".into(),
            target_artifact: "run".into(),
        });
        assert!(validate_artifact_facet_links(&export.graph, &export.context).is_err());
    }
}

fn add_unlinked_outcome(export: &mut ArtifactFacetExport, id: &str, outcome: &TrainingOutcome) {
    let proof = proof(
        FACET_TRAINING_OUTCOME,
        outcome.schema_version,
        &serde_json::to_vec(outcome).unwrap(),
    );
    let mut artifact = export
        .graph
        .artifacts
        .iter()
        .find(|a| a.id == "result")
        .unwrap()
        .clone();
    artifact.id = id.into();
    artifact.identity = proof.identity().clone();
    artifact.storage = ArtifactFacetStorage::Reference {
        root_sha256: Some(proof.raw_sha256().into()),
    };
    export.graph.artifacts.push(artifact);
    export.context.owner_proofs.insert(id.into(), proof);
}

#[test]
fn unrelated_valid_outcomes_cannot_supply_cache_or_recipe_witnesses() {
    let (outcome, _) = source();
    let mut same_bundle = outcome.clone();
    same_bundle.outcome_id = "outcome:other-same-bundle".into();
    same_bundle.outcome_fingerprint = same_bundle.compute_fingerprint().unwrap();
    same_bundle.validate().unwrap();
    assert_eq!(same_bundle.execution_bundle, outcome.execution_bundle);
    let mut different_scores = same_bundle.clone();
    different_scores.outcome_id = "outcome:other-score-context".into();
    different_scores.score_set.reports[0]
        .metrics
        .insert("mae".into(), 0.5);
    different_scores.execution_bundle.scores = Some(different_scores.score_set.clone());
    different_scores.outcome_fingerprint = different_scores.compute_fingerprint().unwrap();
    different_scores.validate().unwrap();
    assert_eq!(
        different_scores.portable_prediction_caches,
        outcome.portable_prediction_caches
    );
    let mut export = export();
    export.graph.links.clear();
    let bundle = export.context.owner_proofs["bundle"].clone();
    replace_proof(&mut export, "run", bundle);
    for (id, relation, source, target) in [
        (
            "cache-run",
            ArtifactFacetRelation::ProducedBy,
            "prediction",
            "run",
        ),
        (
            "graph-bundle",
            ArtifactFacetRelation::RecipeOf,
            "recipe",
            "run",
        ),
        (
            "score-cache",
            ArtifactFacetRelation::Evaluates,
            "score",
            "prediction",
        ),
    ] {
        export.graph.links.push(ArtifactFacetLink {
            id: id.into(),
            owner: "dag-ml".into(),
            relation,
            source_artifact: source.into(),
            target_artifact: target.into(),
        });
    }
    // Start with only direct pair proofs. No outcome witness is required.
    export
        .context
        .owner_proofs
        .retain(|id, _| ["prediction", "recipe", "score", "run"].contains(&id.as_str()));
    let baseline = validate_artifact_facet_links(&export.graph, &export.context).unwrap();
    assert_eq!(
        baseline
            .iter()
            .find(|x| x.link_id == "graph-bundle")
            .unwrap()
            .verdict
            .status,
        FacetSemanticStatus::Validated
    );
    for (first, second) in [("a-unlinked", "z-unlinked"), ("z-unlinked", "a-unlinked")] {
        let mut graph = export.graph.clone();
        let mut proofs = export.context.owner_proofs.clone();
        add_unlinked_outcome(&mut export, first, &same_bundle);
        add_unlinked_outcome(&mut export, second, &different_scores);
        assert_eq!(
            validate_artifact_facet_links(&export.graph, &export.context).unwrap(),
            baseline
        );
        std::mem::swap(&mut graph, &mut export.graph);
        std::mem::swap(&mut proofs, &mut export.context.owner_proofs);
    }
}

struct CountingDatasetPort(std::cell::Cell<usize>);
impl FacetDatasetBindingValidator for CountingDatasetPort {
    fn validate_dataset_binding(
        &self,
        _: &ArtifactFacet,
        _: Option<&[u8]>,
        _: &ValidatedOwnerArtifact,
    ) -> FacetResult<FacetLinkVerdict> {
        self.0.set(self.0.get() + 1);
        Ok(FacetLinkVerdict::unverified(
            "byte integrity alone supplies no IO-to-DMD/DAG mapping proof",
        ))
    }
}

#[test]
fn candidate_dataset_bytes_are_bound_before_any_semantic_port_call() {
    let owner_bytes = b"{\"owner_emitted\":true}".to_vec();
    for embedded in [false, true] {
        let mut export = export();
        let mut dataset = reference("dataset", ArtifactFacetKind::Dataset);
        dataset.identity.fingerprint_profile = "sha256.raw.v1".into();
        dataset.identity.fingerprint = format!("{:x}", sha2::Sha256::digest(&owner_bytes));
        if embedded {
            dataset.storage = ArtifactFacetStorage::Embedded {
                root_member: "io/dataset.json".into(),
                member_paths: vec!["io/dataset.json".into()],
            };
        }
        export.graph.artifacts.push(dataset);
        export.graph.links.push(ArtifactFacetLink {
            id: "dataset".into(),
            owner: "dag-ml".into(),
            relation: ArtifactFacetRelation::DatasetOf,
            source_artifact: "dataset".into(),
            target_artifact: "model".into(),
        });
        let port = CountingDatasetPort(std::cell::Cell::new(0));
        let mut context = export.context.with_dataset_binding_validator(&port);
        context
            .external_bytes
            .insert("dataset".into(), b"wrong candidate".to_vec());
        assert!(validate_artifact_facet_links(&export.graph, &context).is_err());
        assert_eq!(port.0.get(), 0);
        context
            .external_bytes
            .insert("dataset".into(), owner_bytes.clone());
        let report = validate_artifact_facet_links(&export.graph, &context).unwrap();
        assert_eq!(port.0.get(), 1);
        assert_eq!(
            report.last().unwrap().verdict.status,
            FacetSemanticStatus::Unverified
        );
        port.0.set(0);
        context
            .external_bytes
            .insert("undeclared".into(), owner_bytes.clone());
        assert!(validate_artifact_facet_links(&export.graph, &context).is_err());
        assert_eq!(port.0.get(), 0);
        context.external_bytes.remove("undeclared");
        if !embedded {
            export.graph.artifacts.last_mut().unwrap().storage = ArtifactFacetStorage::Reference {
                root_sha256: Some("0".repeat(64)),
            };
            assert!(validate_artifact_facet_links(&export.graph, &context).is_err());
            assert_eq!(port.0.get(), 0);
        }
    }
}

#[test]
fn plan_identity_normalizes_actual_owner_defaults_and_retains_original_bytes() {
    let export = export();
    let original = &export.context.owner_proofs["plan"];
    let mut alternate: serde_json::Value =
        serde_json::from_slice(original.original_bytes()).unwrap();
    let port = &mut alternate["graph_plan"]["graph"]["interface"]["inputs"][0];
    assert_eq!(port["description"], "");
    port.as_object_mut().unwrap().remove("description");
    port["unit_level"] = json!(null);
    let bytes = format!(" \n{alternate}\n");
    ExecutionPlan::from_json(&bytes).unwrap();
    let changed = proof(FACET_EXECUTION_PLAN, 1, bytes.as_bytes());
    assert_eq!(changed.identity(), original.identity());
    assert_eq!(
        changed.identity().fingerprint_profile,
        FACET_EXECUTION_PLAN_PROFILE
    );
    assert_eq!(changed.original_bytes(), bytes.as_bytes());
    assert_ne!(changed.raw_sha256(), original.raw_sha256());
}

#[test]
fn public_refusals_distinguish_operations_content_capabilities_and_mismatches() {
    let mut export = export();
    assert!(matches!(
        require_facet_operation("predict", "model", &[], &export.graph, &export.context),
        Err(FacetRefusal::UnsupportedOperation { .. })
    ));
    let proof = &export.context.owner_proofs["model"];
    assert!(matches!(
        check_facet_hydration_requirements(proof, &BTreeSet::new(), false),
        Err(FacetRefusal::MissingTrust)
    ));
    assert!(matches!(
        check_facet_hydration_requirements(proof, &BTreeSet::new(), true),
        Err(FacetRefusal::MissingCapability { .. })
    ));
    assert!(matches!(
        inspect_facet_model_states(&export.context.owner_proofs["recipe"]),
        Err(FacetRefusal::UnsupportedPayload { .. })
    ));
    export.context.owner_proofs.remove("recipe");
    assert!(matches!(
        require_facet_operation("inspect", "recipe", &[], &export.graph, &export.context),
        Err(FacetRefusal::UnresolvedOwnerContent { .. })
    ));
    export
        .graph
        .artifacts
        .iter_mut()
        .find(|a| a.id == "model")
        .unwrap()
        .identity
        .fingerprint = "0".repeat(64);
    assert!(matches!(
        validate_artifact_facet_links(&export.graph, &export.context),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        validate_owner_facet_bytes(&FacetContentType::new(FACET_GRAPH_SPEC, 1), b"{bad json"),
        Err(FacetRefusal::InvalidPayload { .. })
    ));
}

#[test]
fn current_v2_cache_refuses_omitted_version_and_changed_outcome_bytes() {
    let (outcome, _) = source();
    assert_eq!(outcome.schema_version, 2);
    assert_eq!(
        outcome
            .portable_prediction_caches
            .as_ref()
            .unwrap()
            .schema_version,
        2
    );
    // V2's owner default is V1, so omitting version from a V2-format set is
    // genuinely invalid. New facet normalization does not relax that contract.
    let mut wrong =
        serde_json::to_value(outcome.portable_prediction_caches.as_ref().unwrap()).unwrap();
    wrong.as_object_mut().unwrap().remove("schema_version");
    assert!(matches!(
        validate_owner_facet_bytes(
            &FacetContentType::new(FACET_PREDICTION_CACHES, 2),
            wrong.to_string().as_bytes()
        ),
        Err(FacetRefusal::OwnerValidation { .. })
    ));
    assert!(
        serde_json::from_value::<BundlePredictionCachePayloadSet>(wrong)
            .unwrap()
            .validate()
            .is_err()
    );
    let mut changed_outcome = serde_json::to_value(&outcome).unwrap();
    changed_outcome["portable_prediction_caches"]["caches"][0]["blocks"][0]["values"][0][0] =
        json!(50420);
    assert!(TrainingOutcome::from_json(&changed_outcome.to_string()).is_err());
}

const ROLE_OUTCOME: &str =
    include_str!("fixtures/artifact_facets/native_role/training_outcome.json");
const ROLE_PACKAGE: &str =
    include_str!("fixtures/artifact_facets/native_role/predictor_package.json");
const ROLE_CHILD: &str = include_str!("fixtures/artifact_facets/native_role/child_refit_v3.json");

fn role_source() -> (
    TrainingOutcome,
    PortablePredictorPackage,
    PortableRefitPackageV3,
) {
    (
        TrainingOutcome::from_json(ROLE_OUTCOME).unwrap(),
        PortablePredictorPackage::from_json(ROLE_PACKAGE).unwrap(),
        PortableRefitPackageV3::from_json(ROLE_CHILD).unwrap(),
    )
}

fn update_package_fingerprints(package: &mut PortablePredictorPackage) {
    let mut owner = TrainingOutcome::from_json(ROLE_OUTCOME).unwrap();
    owner.execution_bundle = package.execution_bundle.clone();
    package.training_outcome.execution_bundle_fingerprint =
        owner.execution_bundle_fingerprint().unwrap();
    package.package_fingerprint = package.compute_fingerprint().unwrap();
    package.validate().unwrap();
}

fn add_model_reference(
    export: &mut ArtifactFacetExport,
    id: &str,
    package: &PortablePredictorPackage,
) {
    let proof = proof(
        FACET_PREDICTOR_PACKAGE,
        package.schema_version,
        &serde_json::to_vec(package).unwrap(),
    );
    let mut artifact = export
        .graph
        .artifacts
        .iter()
        .find(|a| a.id == "model")
        .unwrap()
        .clone();
    artifact.id = id.into();
    artifact.identity = proof.identity().clone();
    artifact.storage = ArtifactFacetStorage::Reference {
        root_sha256: Some(proof.raw_sha256().into()),
    };
    export.graph.artifacts.push(artifact);
    export.context.owner_proofs.insert(id.into(), proof);
}

fn refit_verdict(export: &ArtifactFacetExport) -> FacetSemanticStatus {
    let edge = export
        .graph
        .links
        .iter()
        .find(|edge| edge.relation == ArtifactFacetRelation::RefitOf)
        .unwrap();
    inspect_artifact_facet_links(&export.graph, &export.context)
        .unwrap()
        .into_iter()
        .find(|v| v.link_id == edge.id)
        .unwrap()
        .verdict
        .status
}

#[test]
fn actual_owner_child_refit_preserves_distinct_target_and_source_scope() {
    let (parent, package, child) = role_source();
    let export = build_refit_artifact_facets(&child, &parent, &package).unwrap();
    export.validate().unwrap();
    let child_proof = proof(FACET_REFIT_V3, 3, ROLE_CHILD.as_bytes());
    assert_eq!(child_proof.original_bytes(), ROLE_CHILD.as_bytes());
    assert_eq!(refit_verdict(&export), FacetSemanticStatus::Validated);
    assert_ne!(
        parent.training_request_fingerprint,
        child.outcome.target_training_request.request_fingerprint
    );
    assert_ne!(parent.data_identities, child.outcome.data_identities);
    assert!(export
        .graph
        .links
        .iter()
        .filter(|link| link.relation == ArtifactFacetRelation::Evaluates)
        .all(|link| link.target_artifact == "result"));
    assert!(!child
        .outcome
        .execution_bundle
        .raw_artifact_payloads
        .is_empty());
    let mut wrong_parent = parent.clone();
    wrong_parent.outcome_id.push_str("-another-owner-outcome");
    wrong_parent.outcome_fingerprint = wrong_parent.compute_fingerprint().unwrap();
    wrong_parent.validate().unwrap();
    assert!(matches!(
        build_refit_artifact_facets(&child, &wrong_parent, &package),
        Err(FacetRefusal::Mismatch { .. })
    ));
    let mut wrong = export;
    replace_proof(
        &mut wrong,
        "evidence",
        proof(
            FACET_TRAINING_OUTCOME,
            2,
            &serde_json::to_vec(&wrong_parent).unwrap(),
        ),
    );
    assert_eq!(refit_verdict(&wrong), FacetSemanticStatus::Mismatch);
    assert!(matches!(
        wrong.validate(),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn refit_requires_declared_exact_parent_package_without_fabricated_ambiguity() {
    let (parent, package, child) = role_source();
    let mut export = build_refit_artifact_facets(&child, &parent, &package).unwrap();
    let mut alias = package.clone();
    alias.package_id.push_str("-another-export-id");
    alias.package_fingerprint = alias.compute_fingerprint().unwrap();
    alias.validate().unwrap();
    assert_ne!(alias.package_fingerprint, package.package_fingerprint);
    assert_eq!(alias.training_outcome, package.training_outcome);
    add_model_reference(&mut export, "other_export", &alias);
    export.graph.links.push(ArtifactFacetLink {
        id: "other-export-evidence".into(),
        owner: "dag-ml".into(),
        relation: ArtifactFacetRelation::EvidenceFor,
        source_artifact: "evidence".into(),
        target_artifact: "other_export".into(),
    });
    assert_eq!(refit_verdict(&export), FacetSemanticStatus::Validated);
    export.graph.links.retain(|link| {
        !(link.relation == ArtifactFacetRelation::EvidenceFor && link.target_artifact == "model")
    });
    // The exact package proof remains valid and present but is unlinked.
    assert!(export.context.owner_proofs.contains_key("model"));
    assert_eq!(refit_verdict(&export), FacetSemanticStatus::Unverified);
    export
        .graph
        .links
        .retain(|link| link.relation != ArtifactFacetRelation::EvidenceFor);
    assert_eq!(refit_verdict(&export), FacetSemanticStatus::Unverified);
    // A genuinely different validated parent package is conflicting declared
    // evidence, not a fake same-hash/different-identity ambiguity.
    let mut foreign = parent.clone();
    foreign.outcome_id.push_str("-foreign");
    foreign.outcome_fingerprint = foreign.compute_fingerprint().unwrap();
    foreign.validate().unwrap();
    let foreign = foreign
        .to_portable_predictor_package(
            "package:foreign-role",
            FittedArtifactMode::PortableRequired,
            ArtifactLoadMode::NativePortable,
        )
        .unwrap();
    add_model_reference(&mut export, "foreign_model", &foreign);
    export.graph.links.push(ArtifactFacetLink {
        id: "conflicting-evidence".into(),
        owner: "dag-ml".into(),
        relation: ArtifactFacetRelation::EvidenceFor,
        source_artifact: "evidence".into(),
        target_artifact: "foreign_model".into(),
    });
    assert!(matches!(
        validate_artifact_facet_links(&export.graph, &export.context),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn initial_full_refit_cannot_claim_an_actual_cv_parent() {
    let initial =
        InitialFullRefitPackage::from_json(std::str::from_utf8(INITIAL).unwrap()).unwrap();
    let mut export = build_initial_refit_artifact_facets(&initial).unwrap();
    let parent = TrainingOutcome::from_json(ROLE_OUTCOME).unwrap();
    let parent = proof(
        FACET_TRAINING_OUTCOME,
        2,
        &serde_json::to_vec(&parent).unwrap(),
    );
    export.graph.artifacts.push(ArtifactFacet {
        id: "evidence".into(),
        kind: ArtifactFacetKind::TrainingEvidence,
        owner: parent.identity().owner.clone(),
        content_type: parent.content_type().clone(),
        identity: parent.identity().clone(),
        storage: ArtifactFacetStorage::Reference {
            root_sha256: Some(parent.raw_sha256().into()),
        },
        requirements: None,
    });
    export
        .context
        .owner_proofs
        .insert("evidence".into(), parent);
    export.graph.links.push(ArtifactFacetLink {
        id: "invalid-initial-parent".into(),
        owner: "dag-ml".into(),
        relation: ArtifactFacetRelation::RefitOf,
        source_artifact: "full_refit".into(),
        target_artifact: "evidence".into(),
    });
    assert_eq!(refit_verdict(&export), FacetSemanticStatus::Mismatch);
    assert!(matches!(
        export.validate(),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn actual_role_pipeline_closure_resolves_exact_members_and_refuses_mutations() {
    let (outcome, package, _) = role_source();
    let mut export =
        build_training_artifact_facets_from_bytes(ROLE_OUTCOME.as_bytes(), ROLE_PACKAGE.as_bytes())
            .unwrap();
    export.validate().unwrap();
    assert_eq!(export.members["dagml/model.json"], ROLE_PACKAGE.as_bytes());
    let artifact = export
        .graph
        .artifacts
        .iter()
        .find(|a| a.id == "model")
        .unwrap();
    let proof = &export.context.owner_proofs["model"];
    let inventory = validate_facet_state_members(artifact, proof, &export.members).unwrap();
    assert!(!inventory.is_empty());
    for state in &inventory {
        assert!(state
            .member_path
            .ends_with(&format!("/{}.bin", state.sha256)));
        assert_eq!(
            export.members[&state.member_path],
            package.execution_bundle.raw_artifact_payloads[&state.artifact_id]
        );
        assert_eq!(
            state.sha256,
            format!(
                "{:x}",
                sha2::Sha256::digest(&export.members[&state.member_path])
            )
        );
    }
    assert_eq!(
        export.members[&inventory[0].member_path],
        include_bytes!("fixtures/artifact_facets/native_role/source_role_pipeline_0.raw.json")
    );
    let wrapper: serde_json::Value =
        serde_json::from_slice(&export.members[&inventory[0].member_path]).unwrap();
    let opaque_state: Vec<u8> = serde_json::from_value(wrapper["states"][0].clone()).unwrap();
    assert_eq!(
        opaque_state,
        include_bytes!("fixtures/artifact_facets/native_role/source_role_pipeline_0.state_0.n4me")
    );
    export.members.get_mut(&inventory[0].member_path).unwrap()[0] ^= 1;
    assert!(matches!(
        export.validate(),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        validate_facet_state_members(artifact, proof, &export.members),
        Err(FacetRefusal::Mismatch { .. })
    ));
    // This exercises the declarative recipe check after the ordinary owner
    // package structural validator accepts the re-fingerprinted candidate.
    let mut changed = package;
    let id = changed.execution_bundle.refit_artifacts[0]
        .artifact
        .id
        .clone();
    let mut wrapper: serde_json::Value =
        serde_json::from_slice(&changed.execution_bundle.raw_artifact_payloads[&id]).unwrap();
    wrapper["steps"][0]["params"]["n_components"] = json!(2);
    let bytes = serde_json::to_vec(&wrapper).unwrap();
    let raw = format!("{:x}", sha2::Sha256::digest(&bytes));
    let artifact = &mut changed.execution_bundle.refit_artifacts[0].artifact;
    artifact.content_fingerprint = Some(raw.clone());
    artifact.size_bytes = Some(bytes.len() as u64);
    artifact.uri = Some(format!("artifacts/{raw}.json"));
    changed
        .execution_bundle
        .raw_artifact_payloads
        .insert(id, bytes);
    update_package_fingerprints(&mut changed);
    assert!(matches!(
        validate_owner_facet_bytes(
            &FacetContentType::new(FACET_PREDICTOR_PACKAGE, 2),
            &serde_json::to_vec(&changed).unwrap()
        ),
        Err(FacetRefusal::OwnerValidation { .. })
    ));
    outcome.validate().unwrap();
}

#[test]
fn same_sha_states_deduplicate_within_facet_and_resolve_by_artifact_id() {
    let (_, mut package, _) = role_source();
    let original = package.execution_bundle.refit_artifacts[0].clone();
    let mut duplicate = original.clone();
    duplicate.artifact.id = ArtifactId::new("artifact:role:duplicate-state-reference").unwrap();
    let bytes = package.execution_bundle.raw_artifact_payloads[&original.artifact.id].clone();
    package
        .execution_bundle
        .raw_artifact_payloads
        .insert(duplicate.artifact.id.clone(), bytes.clone());
    package
        .execution_bundle
        .refit_artifacts
        .push(duplicate.clone());
    package.artifact_bindings.push(PackageArtifactBinding {
        artifact_id: duplicate.artifact.id,
        load_mode: ArtifactLoadMode::NativePortable,
    });
    package
        .artifact_bindings
        .sort_by(|a, b| a.artifact_id.cmp(&b.artifact_id));
    update_package_fingerprints(&mut package);
    let proof = proof(
        FACET_PREDICTOR_PACKAGE,
        2,
        &serde_json::to_vec(&package).unwrap(),
    );
    let expected = expected_facet_state_members("model", &proof).unwrap();
    assert_eq!(expected.len(), 2);
    assert_eq!(expected[0].member_path, expected[1].member_path);
    let cross_facet = expected_facet_state_members("full_refit", &proof).unwrap();
    assert_ne!(expected[0].member_path, cross_facet[0].member_path);
    let mut reordered = package.clone();
    reordered.execution_bundle.raw_artifact_payloads = package
        .execution_bundle
        .raw_artifact_payloads
        .iter()
        .rev()
        .map(|(id, bytes)| (id.clone(), bytes.clone()))
        .collect();
    update_package_fingerprints(&mut reordered);
    let reordered = proof_for_package(&reordered);
    assert_eq!(
        expected,
        expected_facet_state_members("model", &reordered).unwrap()
    );
    // The same validated package raw map is emitted by the training builder.
    let mut outcome = TrainingOutcome::from_json(ROLE_OUTCOME).unwrap();
    outcome.execution_bundle = package.execution_bundle.clone();
    for lineage in outcome
        .lineage
        .iter_mut()
        .filter(|lineage| lineage.phase == Phase::Refit)
    {
        lineage.artifact_refs = outcome
            .execution_bundle
            .refit_artifacts
            .iter()
            .filter(|record| record.node_id == lineage.node_id)
            .map(|record| record.artifact.clone())
            .collect();
    }
    outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
    outcome.validate().unwrap();
    let package = outcome
        .to_portable_predictor_package(
            "package:deduplicated-role",
            FittedArtifactMode::PortableRequired,
            ArtifactLoadMode::NativePortable,
        )
        .unwrap();
    let export = build_training_artifact_facets(&outcome, &package).unwrap();
    let artifact = export
        .graph
        .artifacts
        .iter()
        .find(|a| a.id == "model")
        .unwrap();
    let ArtifactFacetStorage::Embedded { member_paths, .. } = &artifact.storage else {
        panic!("expected embedded model")
    };
    assert_eq!(member_paths.len(), 2); // root plus one physical state member
    validate_facet_state_members(
        artifact,
        &export.context.owner_proofs["model"],
        &export.members,
    )
    .unwrap();
}

fn proof_for_package(package: &PortablePredictorPackage) -> ValidatedOwnerArtifact {
    proof(
        FACET_PREDICTOR_PACKAGE,
        package.schema_version,
        &serde_json::to_vec(package).unwrap(),
    )
}

#[cfg(feature = "methods-optimizer")]
struct FacetNativeProvider {
    identity: TrainingDataIdentity,
    relations: SampleRelationSet,
    rows: BTreeMap<SampleId, ([f64; 4], f64)>,
    next: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "methods-optimizer")]
impl RuntimeDataProvider for FacetNativeProvider {
    fn materialize(&self, _: &DataMaterializationRequest) -> Result<HandleRef> {
        Ok(HandleRef {
            handle: self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            kind: HandleKind::Data,
            owner_controller: ControllerId::new("provider:facet-native")?,
        })
    }
    fn make_view(&self, _: &DataViewRequest) -> Result<HandleRef> {
        Ok(HandleRef {
            handle: self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            kind: HandleKind::DataView,
            owner_controller: ControllerId::new("provider:facet-native")?,
        })
    }
    fn training_data_identity(&self, _: &DataBinding) -> Result<Option<TrainingDataIdentity>> {
        Ok(Some(self.identity.clone()))
    }
    fn coordinator_relations(&self, _: &DataBinding) -> Result<Option<SampleRelationSet>> {
        Ok(Some(self.relations.clone()))
    }
    fn methods_pls_capability(&self) -> Result<()> {
        Ok(())
    }
    fn methods_pls_data(&self, request: &MethodsPlsDataRequest) -> Result<MethodsPlsData> {
        let dataset = |view: &DataProviderViewSpec, targets: bool| -> Result<MethodsPlsDataset> {
            let ids = view
                .sample_ids
                .clone()
                .unwrap_or_else(|| self.rows.keys().cloned().collect());
            let mut x = Vec::new();
            let mut y = Vec::new();
            for id in &ids {
                let (row, target) = self.rows.get(id).ok_or_else(|| {
                    DagMlError::RuntimeValidation(
                        "native facet test view references an absent provider row".into(),
                    )
                })?;
                x.extend_from_slice(row);
                y.push(*target);
            }
            let count = ids.len();
            Ok(MethodsPlsDataset {
                sample_ids: ids,
                x: MethodsPlsMatrix {
                    values: x,
                    rows: count,
                    cols: 4,
                },
                y: targets.then_some(MethodsPlsMatrix {
                    values: y,
                    rows: count,
                    cols: 1,
                }),
                target_names: vec!["protein".into()],
                y_validity_masks: None,
            })
        };
        Ok(MethodsPlsData {
            fit: dataset(&request.fit_view, request.phase != Phase::Predict)?,
            prediction: request
                .prediction_view
                .as_ref()
                .map(|view| dataset(view, true))
                .transpose()?,
        })
    }
}

#[cfg(feature = "methods-optimizer")]
fn current_facet_runtime() -> MethodsRuntime {
    let path = std::env::var("N4M_LIBRARY_PATH")
        .expect("native facet tests require an actual Methods owner library");
    assert_eq!(std::env::var("N4M_LIB_PATH").unwrap(), path);
    assert_eq!(
        format!("{:x}", sha2::Sha256::digest(std::fs::read(&path).unwrap())),
        std::env::var("N4M_EXPECTED_LIBRARY_SHA256")
            .expect("native test requires the qualified owner library SHA256")
    );
    configure_facet_methods_runtime(path).unwrap()
}

#[cfg(feature = "methods-optimizer")]
fn actual_n4mm3_package(runtime: &MethodsRuntime) -> PortablePredictorPackage {
    let mut request = TrainingRequest::from_json(include_str!(
        "fixtures/artifact_facets/native_role/source_training_request.json"
    ))
    .unwrap();
    let model = request.graph.nodes.first_mut().unwrap();
    model.operator = Some(json!("pls"));
    model.seed_label = None;
    model.params = BTreeMap::from([
        ("n_components".into(), json!(1)),
        (
            "pipeline".into(),
            json!({"schema_version":1,"pipeline_type":"n4m.snv_savgol_smooth.v1","savgol_window":3,"savgol_poly_degree":2}),
        ),
    ]);
    let manifest = request.controller_manifests.first_mut().unwrap();
    manifest.controller_id = ControllerId::new(hpo::METHODS_PLS_CONTROLLER_ID).unwrap();
    manifest.controller_version = "libn4m-2.5".into();
    manifest.operator_selectors =
        serde_json::from_value(json!([{"refs":["pls"],"types":["pls"]}])).unwrap();
    let original_rows: BTreeMap<String, [f64; 4]> = serde_json::from_str(include_str!(
        "fixtures/artifact_facets/native_role/source_provider_rows.json"
    ))
    .unwrap();
    // Provider-owned synthetic test rows. These are never an IO identity or
    // IO-to-DAG projection claim; the native inspector reports the real shape.
    let rows: BTreeMap<SampleId, ([f64; 4], f64)> = original_rows
        .iter()
        .map(|(id, row)| (SampleId::new(id).unwrap(), (*row, row[3])))
        .collect();
    let features: Vec<_> = rows.iter().map(|(id, (x, _))| (id, x)).collect();
    let targets: Vec<_> = rows.iter().map(|(id, (_, y))| (id, y)).collect();
    request.data_identities[0].data_content_fingerprint = format!(
        "{:x}",
        sha2::Sha256::digest(serde_json::to_vec(&features).unwrap())
    );
    request.data_identities[0].target_content_fingerprint = format!(
        "{:x}",
        sha2::Sha256::digest(serde_json::to_vec(&targets).unwrap())
    );
    request.data_identities[0].identity_fingerprint =
        request.data_identities[0].compute_fingerprint().unwrap();
    request.request_fingerprint = request.compute_fingerprint().unwrap();
    request.validate().unwrap();
    let relations: SampleRelationSet = serde_json::from_str(include_str!(
        "fixtures/artifact_facets/native_role/source_relations.json"
    ))
    .unwrap();
    let projection = request.project().unwrap();
    let influence =
        TrainingInfluenceManifest::derive_for_projection(&projection, &request, &relations)
            .unwrap();
    let provider = FacetNativeProvider {
        identity: request.data_identities[0].clone(),
        relations: relations.clone(),
        rows,
        next: std::sync::atomic::AtomicU64::new(1),
    };
    let mut controllers = RuntimeControllerRegistry::new();
    register_methods_native_controllers(&mut controllers, runtime.clone()).unwrap();
    let mut store = InMemoryArtifactStore::new();
    let outcome = execute_training(TrainingExecutionInput {
        request: &request,
        outcome_id: "outcome:facet-n4mm3".into(),
        run_id: RunId::new("run:facet-n4mm3").unwrap(),
        bundle_id: BundleId::new("bundle:facet-n4mm3").unwrap(),
        controllers: &controllers,
        data_provider: &provider,
        relations: &relations,
        training_influence: &influence,
        artifact_store: &mut store,
        warnings: Vec::new(),
        diagnostics: BTreeMap::new(),
    })
    .unwrap();
    outcome
        .to_portable_predictor_package(
            "package:facet-n4mm3",
            FittedArtifactMode::PortableRequired,
            ArtifactLoadMode::NativePortable,
        )
        .unwrap()
}

#[cfg(feature = "methods-optimizer")]
#[test]
fn facet_inspector_reports_actual_n4mm3_and_refuses_structurally_valid_descriptor_mismatch() {
    let runtime = current_facet_runtime();
    let package = actual_n4mm3_package(&runtime);
    let validated = proof_for_package(&package);
    let report = inspect_facet_model_states_with_methods(&validated, &runtime).unwrap();
    assert!(!report.is_empty());
    assert!(report
        .iter()
        .all(|item| item.verdict.status == FacetSemanticStatus::Validated
            && item
                .native_descriptor
                .as_ref()
                .is_some_and(|descriptor| descriptor.format_version == 3)));
    let mut changed = package.clone();
    let descriptor = changed.execution_bundle.refit_artifacts[0]
        .artifact
        .native_predictor_descriptor
        .as_mut()
        .unwrap();
    descriptor.writer_abi.patch += 1;
    descriptor.descriptor_fingerprint = descriptor.compute_fingerprint().unwrap();
    update_package_fingerprints(&mut changed);
    let structurally_valid = proof_for_package(&changed);
    assert!(matches!(
        inspect_facet_model_states_with_methods(&structurally_valid, &runtime),
        Err(FacetRefusal::Mismatch { .. })
    ));
    let (_, role, _) = role_source();
    let role = proof_for_package(&role);
    let opaque = inspect_facet_model_states_with_methods(&role, &runtime).unwrap();
    assert!(!opaque.is_empty());
    assert!(opaque.iter().all(
        |item| item.verdict.status == FacetSemanticStatus::Unverified
            && item.native_descriptor.is_none()
    ));
    // The OnceLock path cannot switch, even while a Context exists. This is
    // a real refusal check, not a successful model-execution qualification.
    let _context = n4m::Context::new().unwrap();
    configure_facet_methods_runtime(runtime.library_path()).unwrap();
    assert!(matches!(
        configure_facet_methods_runtime(std::env::current_exe().unwrap()),
        Err(FacetRefusal::RuntimeUnavailable { .. })
    ));
    assert!(matches!(
        configure_facet_methods_runtime("/tmp/absent-facet-methods-runtime.so"),
        Err(FacetRefusal::RuntimeUnavailable { .. })
    ));
}

fn genuine_role_export() -> ArtifactFacetExport {
    build_training_artifact_facets_from_bytes(ROLE_OUTCOME.as_bytes(), ROLE_PACKAGE.as_bytes())
        .unwrap()
}

fn append_physical_member(export: &mut ArtifactFacetExport, id: &str, path: &str, bytes: Vec<u8>) {
    let artifact = export
        .graph
        .artifacts
        .iter_mut()
        .find(|artifact| artifact.id == id)
        .unwrap();
    let ArtifactFacetStorage::Embedded { member_paths, .. } = &mut artifact.storage else {
        panic!("expected embedded owner facet")
    };
    member_paths.push(path.into());
    assert!(export.members.insert(path.into(), bytes).is_none());
}

#[test]
fn every_genuine_proven_non_state_facet_refuses_smuggled_support_members() {
    for id in ["result", "score", "recipe", "bundle", "plan", "prediction"] {
        let mut export = genuine_role_export();
        export.validate().unwrap();
        let artifact = export.graph.artifacts.iter().find(|a| a.id == id).unwrap();
        assert!(validate_facet_members(
            artifact,
            &export.context.owner_proofs[id],
            &export.members
        )
        .unwrap()
        .is_empty());
        append_physical_member(
            &mut export,
            id,
            &format!("smuggled/{id}.bin"),
            b"not owner bytes".to_vec(),
        );
        let artifact = export.graph.artifacts.iter().find(|a| a.id == id).unwrap();
        assert!(
            matches!(
                validate_facet_members(artifact, &export.context.owner_proofs[id], &export.members),
                Err(FacetRefusal::Mismatch { .. })
            ),
            "reader accepted {id}"
        );
        assert!(
            matches!(export.validate(), Err(FacetRefusal::Mismatch { .. })),
            "export accepted {id}"
        );
    }
}

#[test]
fn actual_model_wrapper_cannot_supply_raw_closure_to_result_attachment_or_run() {
    let mut export = genuine_role_export();
    let model = export
        .graph
        .artifacts
        .iter()
        .find(|a| a.id == "model")
        .unwrap();
    let expected = validate_facet_members(
        model,
        &export.context.owner_proofs["model"],
        &export.members,
    )
    .unwrap();
    assert!(!expected.is_empty());
    let raw = export.members[&expected[0].member_path].clone();
    assert_eq!(
        raw,
        include_bytes!("fixtures/artifact_facets/native_role/source_role_pipeline_0.raw.json")
    );
    let wrong = format!(
        "dagml/facets/{}/states/{}.bin",
        "0".repeat(64),
        "f".repeat(64)
    );
    append_physical_member(&mut export, "result", &wrong, raw);
    let result = export
        .graph
        .artifacts
        .iter()
        .find(|a| a.id == "result")
        .unwrap();
    for kind in [
        ArtifactFacetKind::Result,
        ArtifactFacetKind::Attachment,
        ArtifactFacetKind::Run,
    ] {
        let mut descriptor = result.clone();
        descriptor.kind = kind;
        // Outcome legitimately supports Result and Run; Attachment is storage
        // only. None may claim external states merely because its body has raw.
        assert!(matches!(
            validate_facet_members(
                &descriptor,
                &export.context.owner_proofs["result"],
                &export.members
            ),
            Err(FacetRefusal::Mismatch { .. })
        ));
        assert!(matches!(
            validate_facet_state_members(
                &descriptor,
                &export.context.owner_proofs["result"],
                &export.members
            ),
            Err(FacetRefusal::Mismatch { .. })
        ));
        let ArtifactFacetStorage::Embedded {
            root_member,
            member_paths,
        } = &mut descriptor.storage
        else {
            unreachable!()
        };
        *member_paths = vec![root_member.clone()];
        assert!(validate_facet_members(
            &descriptor,
            &export.context.owner_proofs["result"],
            &export.members
        )
        .unwrap()
        .is_empty());
    }
    assert!(matches!(
        export.validate(),
        Err(FacetRefusal::Mismatch { .. })
    ));
    // Descriptor/proof agreement is checked before physical closure, including
    // when the root bytes themselves would otherwise be acceptable.
    let mut wrong_descriptor = result.clone();
    wrong_descriptor.content_type.version += 1;
    assert!(matches!(
        validate_facet_members(
            &wrong_descriptor,
            &export.context.owner_proofs["result"],
            &BTreeMap::new()
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn opaque_unproven_embedded_content_does_not_become_owner_proof() {
    let mut export = genuine_role_export();
    let mut opaque = reference("future-attachment", ArtifactFacetKind::Attachment);
    opaque.storage = ArtifactFacetStorage::Embedded {
        root_member: "future/root.bin".into(),
        member_paths: vec!["future/root.bin".into(), "future/support.bin".into()],
    };
    export.graph.artifacts.push(opaque);
    export.members.insert("future/root.bin".into(), vec![1]);
    export.members.insert("future/support.bin".into(), vec![2]);
    export.validate().unwrap(); // physical storage only; no owner proof exists
    assert!(matches!(
        require_facet_operation(
            "inspect",
            "future-attachment",
            &[],
            &export.graph,
            &export.context
        ),
        Err(FacetRefusal::UnresolvedOwnerContent { .. })
    ));
    export.context.owner_proofs.remove("model");
    assert!(matches!(
        export.validate(),
        Err(FacetRefusal::UnresolvedOwnerContent { .. })
    ));
}

fn assert_cross_run_mismatch(mut export: ArtifactFacetExport, source: &str, target: &str) {
    let id = "adversarial-produced-by";
    export.graph.links.push(ArtifactFacetLink {
        id: id.into(),
        owner: "dag-ml".into(),
        relation: ArtifactFacetRelation::ProducedBy,
        source_artifact: source.into(),
        target_artifact: target.into(),
    });
    let report = inspect_artifact_facet_links(&export.graph, &export.context).unwrap();
    assert_eq!(
        report
            .iter()
            .find(|link| link.link_id == id)
            .unwrap()
            .verdict
            .status,
        FacetSemanticStatus::Mismatch,
        "{source} -> {target}"
    );
    assert!(matches!(
        validate_artifact_facet_links(&export.graph, &export.context),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        export.validate(),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn live_parent_and_child_proofs_refuse_cross_run_producer_claims() {
    let (parent, package, child) = role_source();
    assert_ne!(parent.run_id, child.outcome.run_id);
    for source in ["score", "model", "result", "prediction", "evidence"] {
        let export = build_refit_artifact_facets(&child, &parent, &package).unwrap();
        export.validate().unwrap();
        assert_cross_run_mismatch(export, source, "refit_run");
    }
    let export = build_refit_artifact_facets(&child, &parent, &package).unwrap();
    assert_cross_run_mismatch(export, "full_refit", "run");
}

fn child_export_with_initial_references() -> ArtifactFacetExport {
    let (parent, package, child) = role_source();
    let initial =
        InitialFullRefitPackage::from_json(std::str::from_utf8(INITIAL).unwrap()).unwrap();
    assert_ne!(initial.run_id, parent.run_id);
    assert_ne!(initial.run_id, child.outcome.run_id);
    let initial = build_initial_refit_artifact_facets(&initial).unwrap();
    initial.validate().unwrap();
    let mut export = build_refit_artifact_facets(&child, &parent, &package).unwrap();
    for (source_id, id) in [
        ("run", "initial_run"),
        ("full_refit", "initial_refit"),
        ("model", "initial_model"),
    ] {
        let proof = initial.context.owner_proofs[source_id].clone();
        let mut artifact = initial
            .graph
            .artifacts
            .iter()
            .find(|a| a.id == source_id)
            .unwrap()
            .clone();
        artifact.id = id.into();
        artifact.storage = ArtifactFacetStorage::Reference {
            root_sha256: Some(proof.raw_sha256().into()),
        };
        export.graph.artifacts.push(artifact);
        export.context.owner_proofs.insert(id.into(), proof);
    }
    export
}

#[test]
fn genuine_initial_refit_run_accepts_its_package_and_refuses_foreign_sources() {
    for source in ["initial_refit", "initial_model"] {
        let mut export = child_export_with_initial_references();
        export.graph.links.push(ArtifactFacetLink {
            id: "same-initial-package".into(),
            owner: "dag-ml".into(),
            relation: ArtifactFacetRelation::ProducedBy,
            source_artifact: source.into(),
            target_artifact: "initial_run".into(),
        });
        let report = export.validate().unwrap();
        assert_eq!(
            report
                .iter()
                .find(|link| link.link_id == "same-initial-package")
                .unwrap()
                .verdict
                .status,
            FacetSemanticStatus::Validated
        );
    }
    for source in [
        "score",
        "model",
        "result",
        "prediction",
        "evidence",
        "full_refit",
    ] {
        assert_cross_run_mismatch(
            child_export_with_initial_references(),
            source,
            "initial_run",
        );
    }
    for (source, target) in [
        ("initial_refit", "run"),
        ("initial_model", "run"),
        ("initial_refit", "refit_run"),
        ("initial_model", "refit_run"),
    ] {
        assert_cross_run_mismatch(child_export_with_initial_references(), source, target);
    }
}

#[test]
fn reference_candidate_root_sha_disagreement_is_a_typed_mismatch() {
    let mut export = genuine_role_export();
    let mut candidate = reference("dataset", ArtifactFacetKind::Dataset);
    candidate.storage = ArtifactFacetStorage::Reference {
        root_sha256: Some("a".repeat(64)),
    };
    export.graph.artifacts.push(candidate);
    export
        .context
        .external_bytes
        .insert("dataset".into(), b"candidate".to_vec());
    assert!(matches!(
        inspect_artifact_facet_links(&export.graph, &export.context),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        validate_artifact_facet_links(&export.graph, &export.context),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn declared_content_version_disagreement_is_structural() {
    let (_, package, _) = role_source();
    assert_eq!(package.schema_version, 2);
    assert!(matches!(
        validate_owner_facet_bytes(
            &FacetContentType::new(FACET_PREDICTOR_PACKAGE, 1),
            ROLE_PACKAGE.as_bytes()
        ),
        Err(FacetRefusal::InvalidStructure { .. })
    ));
}

#[test]
fn refit_packages_cannot_be_produced_by_cv_bundle_but_matching_score_is_unverified() {
    for source in ["full_refit", "initial_refit"] {
        let mut export = child_export_with_initial_references();
        export
            .graph
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.id == "bundle")
            .unwrap()
            .kind = ArtifactFacetKind::Run;
        export.validate().unwrap();
        assert_cross_run_mismatch(export, source, "bundle");
    }

    let mut export = child_export_with_initial_references();
    export
        .graph
        .artifacts
        .iter_mut()
        .find(|artifact| artifact.id == "bundle")
        .unwrap()
        .kind = ArtifactFacetKind::Run;
    export.graph.links.push(ArtifactFacetLink {
        id: "matching-score-by-cv-bundle".into(),
        owner: "dag-ml".into(),
        relation: ArtifactFacetRelation::ProducedBy,
        source_artifact: "score".into(),
        target_artifact: "bundle".into(),
    });
    let report = inspect_artifact_facet_links(&export.graph, &export.context).unwrap();
    assert_eq!(
        report
            .iter()
            .find(|link| link.link_id == "matching-score-by-cv-bundle")
            .unwrap()
            .verdict
            .status,
        FacetSemanticStatus::Unverified
    );
    validate_artifact_facet_links(&export.graph, &export.context).unwrap();
    export.validate().unwrap();
}

#[test]
fn native_role_capture_exactly_matches_direct_owner_producer_and_validates_members() {
    const GRAPH: &[u8] = include_bytes!("fixtures/artifact_facets/native_role/capture/facets.json");
    const INVENTORY: &[u8] =
        include_bytes!("fixtures/artifact_facets/native_role/capture/inventory.json");
    let mut fresh =
        build_training_artifact_facets_from_bytes(ROLE_OUTCOME.as_bytes(), ROLE_PACKAGE.as_bytes())
            .unwrap();
    fresh.validate().unwrap();
    assert_eq!(serde_json::to_vec_pretty(&fresh.graph).unwrap(), GRAPH);
    let committed_graph =
        ArtifactFacetGraph::from_json(std::str::from_utf8(GRAPH).unwrap()).unwrap();
    let committed_members: BTreeMap<String, Vec<u8>> = [
        (
            "dagml/facets/9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4/states/6b30609253276c3da627257ada78ad1ee1eaa8ffff70515fc491c30e9d666905.bin",
            include_bytes!("fixtures/artifact_facets/native_role/capture/dagml/facets/9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4/states/6b30609253276c3da627257ada78ad1ee1eaa8ffff70515fc491c30e9d666905.bin").as_slice(),
        ),
        (
            "dagml/model.json",
            include_bytes!("fixtures/artifact_facets/native_role/capture/dagml/model.json").as_slice(),
        ),
        (
            "dagml/plan.json",
            include_bytes!("fixtures/artifact_facets/native_role/capture/dagml/plan.json").as_slice(),
        ),
        (
            "dagml/prediction.json",
            include_bytes!("fixtures/artifact_facets/native_role/capture/dagml/prediction.json").as_slice(),
        ),
        (
            "dagml/recipe.json",
            include_bytes!("fixtures/artifact_facets/native_role/capture/dagml/recipe.json").as_slice(),
        ),
        (
            "dagml/result.json",
            include_bytes!("fixtures/artifact_facets/native_role/capture/dagml/result.json").as_slice(),
        ),
        (
            "dagml/run.json",
            include_bytes!("fixtures/artifact_facets/native_role/capture/dagml/run.json").as_slice(),
        ),
        (
            "dagml/score.json",
            include_bytes!("fixtures/artifact_facets/native_role/capture/dagml/score.json").as_slice(),
        ),
    ]
    .into_iter()
    .map(|(path, bytes)| (path.into(), bytes.to_vec()))
    .collect();
    assert_eq!(fresh.members, committed_members);
    let inventory = fresh
        .members
        .iter()
        .map(|(path, bytes)| {
            (
                path,
                json!({"byte_length": bytes.len(), "sha256": format!("{:x}", sha2::Sha256::digest(bytes))}),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(serde_json::to_vec_pretty(&inventory).unwrap(), INVENTORY);

    // Stored links.json is diagnostic only. Recreate validation using live
    // owner proofs and every exact committed member, never a stored verdict.
    fresh.graph = committed_graph;
    fresh.members = committed_members;
    fresh.validate().unwrap();
    for artifact in &fresh.graph.artifacts {
        if matches!(artifact.storage, ArtifactFacetStorage::Embedded { .. }) {
            let proof = &fresh.context.owner_proofs[&artifact.id];
            let states = validate_facet_members(artifact, proof, &fresh.members).unwrap();
            if artifact.kind == ArtifactFacetKind::Model {
                assert_eq!(states.len(), 1);
                assert_eq!(
                    fresh.members[&states[0].member_path],
                    include_bytes!(
                        "fixtures/artifact_facets/native_role/source_role_pipeline_0.raw.json"
                    )
                );
            } else {
                assert!(states.is_empty());
            }
        }
    }
}

// Exact Core combined producer manifest, SHA-256
// 4b2cd88270436433955fc9ba8f4851934c4079f534452cdc0c195d6aa5bafa4d.
fn combined_graph() -> ArtifactFacetGraph {
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/artifact_facets/combined-manifest.json"
    ))
    .unwrap();
    serde_json::from_value(json!({"artifacts": manifest["artifacts"], "links": manifest["links"]}))
        .unwrap()
}

#[test]
fn entrypoint_roles_validate_structural_contracts() {
    let manifest: serde_json::Value = serde_json::from_str(include_str!(
        "fixtures/artifact_facets/combined-manifest.json"
    ))
    .unwrap();
    let entry = &manifest["entrypoints"][0];
    let graph = combined_graph();
    let contract = FacetDatasetContract {
        owner: "nirs4all-io",
        content_types: &[("nirs4all.dataset.v1", 1), ("nirs4all.dataset.v2", 2)],
    };
    let inputs: Vec<String> = serde_json::from_value(entry["input_artifacts"].clone()).unwrap();
    let resolve = |op: &str,
                   root: &str,
                   inputs: &[String],
                   req: Option<&str>,
                   graph: &ArtifactFacetGraph,
                   contract: &FacetDatasetContract<'_>| {
        resolve_facet_entrypoint("dag-ml", op, root, inputs, req, graph, contract)
    };
    assert_eq!(
        resolve(
            entry["operation"].as_str().unwrap(),
            entry["root_artifact"].as_str().unwrap(),
            &inputs,
            None,
            &graph,
            &contract
        )
        .unwrap(),
        FacetEntrypointRole::StoredPredict {
            root_id: "model".into(),
            input_ids: vec!["dataset".into()]
        }
    );
    let mut v2 = graph.clone();
    let data = v2.artifacts.iter_mut().find(|a| a.id == "dataset").unwrap();
    data.content_type = FacetContentType::new("nirs4all.dataset.v2", 2);
    assert!(resolve("predict", "model", &inputs, None, &v2, &contract).is_ok());
    for (owner, id, version) in [
        ("foreign", "nirs4all.dataset.v1", 1),
        ("dag-ml", "nirs4all.dataset.v1", 1),
        ("nirs4all-io", "nirs4all.dataset.v1", 2),
    ] {
        let mut g = graph.clone();
        let d = g.artifacts.iter_mut().find(|a| a.id == "dataset").unwrap();
        d.owner = owner.into();
        d.identity.owner = owner.into();
        d.content_type = FacetContentType::new(id, version);
        assert!(matches!(
            resolve("predict", "model", &inputs, None, &g, &contract),
            Err(FacetRefusal::Mismatch { .. })
        ));
    }
    for bad in [
        FacetDatasetContract {
            owner: "nirs4all-io",
            content_types: &[],
        },
        FacetDatasetContract {
            owner: "dag-ml",
            content_types: &[("x", 1)],
        },
    ] {
        assert!(matches!(
            resolve("predict", "model", &inputs, None, &graph, &bad),
            Err(FacetRefusal::InvalidStructure { .. })
        ));
    }
    assert!(matches!(
        resolve("predict", "model", &[], None, &graph, &contract),
        Err(FacetRefusal::UnsupportedOperation { .. })
    ));
    let mut g = graph.clone();
    let mut second = g
        .artifacts
        .iter()
        .find(|a| a.id == "dataset")
        .unwrap()
        .clone();
    second.id = "dataset2".into();
    second.storage = ArtifactFacetStorage::Embedded {
        root_member: "io/second".into(),
        member_paths: vec!["io/second".into()],
    };
    g.artifacts.push(second);
    assert!(matches!(
        resolve(
            "predict",
            "model",
            &["dataset".into(), "dataset2".into()],
            None,
            &g,
            &contract
        ),
        Err(FacetRefusal::UnsupportedOperation { .. })
    ));
    assert!(matches!(
        resolve(
            "predict",
            "model",
            &["dataset".into(), "dataset".into()],
            None,
            &g,
            &contract
        ),
        Err(FacetRefusal::InvalidStructure { .. })
    ));
    assert!(matches!(
        resolve(
            "predict",
            "model",
            &inputs,
            Some("dataset"),
            &graph,
            &contract
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        resolve(
            "predict",
            "model",
            &["missing".into()],
            None,
            &graph,
            &contract
        ),
        Err(FacetRefusal::InvalidStructure { .. })
    ));
    for op in ["Predict", "predict ", "inspect", "score-inspect"] {
        assert!(matches!(
            resolve(op, "model", &[], None, &graph, &contract),
            Err(FacetRefusal::UnsupportedOperation { .. })
        ));
    }
    assert!(matches!(
        resolve_facet_entrypoint("foreign", "predict", "model", &inputs, None, &graph, &contract),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        resolve(
            "predict-external",
            "model",
            &inputs,
            None,
            &graph,
            &contract
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert_eq!(
        resolve(
            "predict-external",
            "model",
            &[],
            Some("dataset"),
            &graph,
            &contract
        )
        .unwrap(),
        FacetEntrypointRole::ExternalPredict {
            root_id: "model".into(),
            schema_anchor: Some("dataset".into())
        }
    );
    assert!(resolve("predict-external", "model", &[], None, &graph, &contract).is_ok());
    assert_eq!(
        resolve(
            "calibrate-external",
            "model",
            &[],
            Some("result"),
            &graph,
            &contract
        )
        .unwrap(),
        FacetEntrypointRole::ExternalCalibrate {
            model_id: "model".into(),
            outcome_id: "result".into()
        }
    );
    assert!(matches!(
        resolve("calibrate-external", "model", &[], None, &graph, &contract),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        resolve(
            "calibrate-external",
            "model",
            &[],
            Some("dataset"),
            &graph,
            &contract
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        resolve("predict", "result", &inputs, None, &graph, &contract),
        Err(FacetRefusal::Mismatch { .. })
    ));
    for id in ["model", "dataset"] {
        let mut g = graph.clone();
        g.artifacts.iter_mut().find(|a| a.id == id).unwrap().storage =
            ArtifactFacetStorage::Reference { root_sha256: None };
        let result = if id == "model" {
            resolve("predict", "model", &inputs, None, &g, &contract)
        } else {
            resolve(
                "predict-external",
                "model",
                &[],
                Some("dataset"),
                &g,
                &contract,
            )
        };
        assert!(matches!(
            result,
            Err(FacetRefusal::UnresolvedOwnerContent { .. })
        ));
    }
    let mut g = graph.clone();
    let model = g.artifacts.iter_mut().find(|a| a.id == "model").unwrap();
    model.owner = "foreign".into();
    model.identity.owner = "foreign".into();
    assert!(matches!(
        resolve("predict", "model", &inputs, None, &g, &contract),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn entrypoint_root_requirement_is_independent_and_full_refit_is_predictable() {
    let mut graph = combined_graph();
    graph.links.clear();
    let contract = FacetDatasetContract {
        owner: "nirs4all-io",
        content_types: &[("nirs4all.dataset.v1", 1)],
    };
    let root = graph
        .artifacts
        .iter_mut()
        .find(|a| a.id == "model")
        .unwrap();
    root.kind = ArtifactFacetKind::FullRefit;
    root.requirements = Some(FacetRequirementReference {
        artifact_id: "result".into(),
    });
    let input = vec!["dataset".into()];
    assert!(resolve_facet_entrypoint(
        "dag-ml", "predict", "model", &input, None, &graph, &contract
    )
    .is_ok());
    assert!(matches!(
        resolve_facet_entrypoint(
            "dag-ml",
            "calibrate-external",
            "model",
            &[],
            Some("result"),
            &graph,
            &contract
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
    assert!(matches!(
        resolve_facet_entrypoint(
            "dag-ml",
            "predict",
            "model",
            &["score".into()],
            None,
            &graph,
            &contract
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

// Immutable run02 capture from the public synthetic parity oracle; see provenance.json.
const JS_ROLE_V2_INITIAL: &[u8] =
    include_bytes!("fixtures/artifact_facets/js_role_v2/initial_refit.json");

fn js_role_v2_source() -> (InitialFullRefitPackage, serde_json::Value) {
    let package =
        InitialFullRefitPackage::from_json(std::str::from_utf8(JS_ROLE_V2_INITIAL).unwrap())
            .unwrap();
    let id = &package.artifacts[0].record.artifact.id;
    let wrapper = serde_json::from_slice(&package.raw_artifact_payloads[id]).unwrap();
    (package, wrapper)
}

// Negative copies bind transport bytes and use the existing owner plan/package
// fingerprint APIs. The original capture and opaque state bytes are never fitted.
fn reseal_js_role_v2_candidate(
    package: &mut InitialFullRefitPackage,
    wrapper: &mut serde_json::Value,
    rebuild_plan: bool,
) {
    if rebuild_plan {
        let plan = &package.effective_plan;
        let mut registry = ControllerRegistry::new();
        for manifest in plan.controller_manifests.values() {
            registry.register(manifest.clone()).unwrap();
        }
        let rebuilt = build_execution_plan(
            plan.id.clone(),
            plan.graph_plan.graph.clone(),
            plan.campaign.clone(),
            &registry,
        )
        .unwrap();
        let node = &rebuilt.node_plans[&package.artifacts[0].record.node_id];
        let record = &mut package.artifacts[0].record;
        record.controller_id = node.controller_id.clone();
        record.artifact.controller_id = node.controller_id.clone();
        record.params_fingerprint = node.params_fingerprint.clone();
        wrapper["params_fingerprint"] = json!(node.params_fingerprint);
        package.effective_plan_fingerprint =
            parse_typed_json(&serde_json::to_string(&rebuilt).unwrap())
                .unwrap()
                .fingerprint()
                .unwrap();
        package.effective_plan = rebuilt;
    }
    let bytes = serde_json::to_vec(wrapper).unwrap();
    let raw = format!("{:x}", sha2::Sha256::digest(&bytes));
    let artifact = &mut package.artifacts[0].record.artifact;
    artifact.content_fingerprint = Some(raw.clone());
    artifact.size_bytes = Some(bytes.len() as u64);
    artifact.uri = Some(format!("artifacts/{raw}.json"));
    package
        .raw_artifact_payloads
        .insert(artifact.id.clone(), bytes);
    package.package_fingerprint = package.compute_fingerprint().unwrap();
    package.validate().unwrap();
}

#[test]
fn genuine_js_v2_initial_refit_facets_preserve_bytes_and_bind_recipe() {
    assert_eq!(
        format!("{:x}", sha2::Sha256::digest(JS_ROLE_V2_INITIAL)),
        "20552dc6779a742f8cddc02a13cf7c3c14056144054fc516a461a4c1def3a918"
    );
    let (package, wrapper) = js_role_v2_source();
    let validated = proof(FACET_INITIAL_REFIT, 1, JS_ROLE_V2_INITIAL);
    assert_eq!(validated.original_bytes(), JS_ROLE_V2_INITIAL);
    assert_eq!(
        validated.identity().fingerprint,
        "27bf09eb9aa7bf96e6c72cea45690d75a96b13d6967f3500cdeeaa67668b42ab"
    );
    assert_eq!(package.package_id, "package:web-holdout");
    assert_eq!(package.run_id.as_str(), "run:web-holdout-refit");
    assert_eq!(package.variant_id.as_str(), "variant:base");
    assert_eq!(
        package
            .training_sample_ids
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        [
            "p3", "p20", "p12", "p27", "p8", "p30", "p15", "p25", "p33", "p37", "p0", "p17", "p6",
            "p10", "p23", "p28", "p26", "p29", "p11", "p31", "p4", "p14", "p32", "p13", "p39",
            "p22", "p35", "p1",
        ]
    );
    let source_s = package
        .training_envelope
        .relation_fingerprint
        .as_deref()
        .unwrap();
    let canonical_c = package.training_relations.fingerprint().unwrap();
    assert_eq!(
        source_s,
        "2395c9d0ee44f21670136244ec9fcd4aac321755d83b67e2575d5176b234a257"
    );
    assert_eq!(
        canonical_c,
        "10404fec25008afabeff39ca5e7ba036bff93eecbcf0a4635bcbdf9ed93b8179"
    );
    assert_ne!(source_s, canonical_c);
    assert_eq!(
        package.training_envelope.coordinator_relations.as_ref(),
        Some(&package.training_relations)
    );
    let record = &package.artifacts[0].record;
    let node = &package.effective_plan.node_plans[&record.node_id];
    let base = &node.params["pipeline_steps_v1"];
    assert_eq!(base[2]["params"]["n_components"], json!(2));
    assert_eq!(node.params["n_components"], json!(6));
    assert_eq!(wrapper["steps"][2]["params"]["n_components"], json!(6));
    assert_eq!(wrapper["steps"][0], base[0]);
    assert_eq!(wrapper["steps"][1], base[1]);
    let mut selected = base.clone();
    selected[2]["params"]["n_components"] = json!(6);
    assert_eq!(wrapper["steps"], selected);
    let raw = &package.raw_artifact_payloads[&record.artifact.id];
    assert_eq!(
        format!("{:x}", sha2::Sha256::digest(raw)),
        "2953f40d1e132f225a10baef73842649eed82d57560bff329cfb6313f49485e9"
    );
    let export = build_initial_refit_artifact_facets(&package).unwrap();
    export.validate().unwrap();
    let exported: InitialFullRefitPackage =
        serde_json::from_slice(&export.members["dagml/full_refit.json"]).unwrap();
    assert_eq!(exported, package);
    let full_refit = export
        .graph
        .artifacts
        .iter()
        .find(|a| a.id == "full_refit")
        .unwrap();
    let members = validate_facet_state_members(
        full_refit,
        &export.context.owner_proofs["full_refit"],
        &export.members,
    )
    .unwrap();
    assert_eq!(members.len(), 1);
    assert_eq!(&export.members[&members[0].member_path], raw);
    assert_eq!(
        export.context.owner_proofs["full_refit"].identity(),
        validated.identity()
    );
}

#[test]
fn js_v2_role_facets_refuse_profile_and_recipe_mismatches() {
    for (case, expected) in [
        ("v2-under-v1-plugin", "unknown field"),
        (
            "v1-schema-under-v2-plugin",
            "exact WASM 1.1.0 pipeline_steps.v1 profile",
        ),
        ("v1-wrapper-under-v2-plugin", "missing field"),
        (
            "unknown-plugin-version",
            "trusted Methods RolePipeline RAW codec",
        ),
        ("other-host-v2", "trusted Methods RolePipeline RAW codec"),
        ("wrong-controller", "trusted Methods RolePipeline RAW codec"),
        (
            "wrong-node-version",
            "exact effective WASM 1.1.0 controller",
        ),
        (
            "wrong-wrapper-version",
            "exact WASM 1.1.0 pipeline_steps.v1 profile",
        ),
        (
            "wrong-profile",
            "exact WASM 1.1.0 pipeline_steps.v1 profile",
        ),
        ("unknown-wrapper-field", "unknown field"),
        (
            "missing-node-recipe",
            "requires explicit node.params.pipeline_steps_v1",
        ),
        (
            "operator-only-recipe",
            "requires explicit node.params.pipeline_steps_v1",
        ),
        ("empty-node-recipe", "needs a nonempty recipe"),
        (
            "conflicting-operator-recipe",
            "operator.steps differs from its base",
        ),
        (
            "saved-selected-param",
            "saved recipe differs from its effective graph",
        ),
        (
            "saved-preprocessing-param",
            "saved recipe differs from its effective graph",
        ),
        ("ambiguous-step", "steps need one nonempty class/methodId"),
        ("null-step-params", "steps need one nonempty class/methodId"),
        (
            "nested-reserved-key",
            "steps need one nonempty class/methodId",
        ),
    ] {
        let (mut package, mut wrapper) = js_role_v2_source();
        let mut rebuild_plan = false;
        match case {
            "v2-under-v1-plugin" => {
                package.artifacts[0].record.artifact.plugin_version = Some("1.0.0".into());
            }
            "v1-schema-under-v2-plugin" => {
                wrapper["schema"] = json!("dagml.methods.regression.v1");
            }
            "v1-wrapper-under-v2-plugin" => {
                wrapper["schema"] = json!("dagml.methods.regression.v1");
                let fields = wrapper.as_object_mut().unwrap();
                fields.remove("controller_version");
                fields.remove("recipe_profile");
            }
            "unknown-plugin-version" => {
                package.artifacts[0].record.artifact.plugin_version = Some("1.2.0".into());
            }
            "other-host-v2" => {
                package.artifacts[0].record.artifact.plugin =
                    Some("dagml.methods.python.regression".into());
            }
            "wrong-controller" | "wrong-node-version" => {
                let manifests = &mut package.effective_plan.controller_manifests;
                let id = package.artifacts[0].record.controller_id.clone();
                let mut manifest = manifests.remove(&id).unwrap();
                if case == "wrong-controller" {
                    manifest.controller_id = ControllerId::new("controller:other").unwrap();
                } else {
                    manifest.controller_version = "1.0.0".into();
                }
                manifests.insert(manifest.controller_id.clone(), manifest);
                rebuild_plan = true;
            }
            "wrong-wrapper-version" => wrapper["controller_version"] = json!("1.0.0"),
            "wrong-profile" => wrapper["recipe_profile"] = json!("other"),
            "unknown-wrapper-field" => wrapper["unknown"] = json!(true),
            "missing-node-recipe"
            | "operator-only-recipe"
            | "empty-node-recipe"
            | "conflicting-operator-recipe" => {
                let node = &mut package.effective_plan.graph_plan.graph.nodes[0];
                let base = node.params["pipeline_steps_v1"].clone();
                match case {
                    "missing-node-recipe" => {
                        node.params.remove("pipeline_steps_v1");
                    }
                    "operator-only-recipe" => {
                        node.operator.as_mut().unwrap()["steps"] = base;
                        node.params.remove("pipeline_steps_v1");
                    }
                    "empty-node-recipe" => {
                        node.params.insert("pipeline_steps_v1".into(), json!([]));
                    }
                    "conflicting-operator-recipe" => {
                        node.operator.as_mut().unwrap()["steps"] = base;
                        node.operator.as_mut().unwrap()["steps"][2]["params"]["n_components"] =
                            json!(6);
                    }
                    _ => unreachable!(),
                }
                rebuild_plan = true;
            }
            "saved-selected-param" => wrapper["steps"][2]["params"]["n_components"] = json!(2),
            "saved-preprocessing-param" => wrapper["steps"][1]["params"]["deriv"] = json!(1),
            "ambiguous-step" => {
                wrapper["steps"][0]["class"] = json!("n4m:preprocessing.scatter.snv");
            }
            "null-step-params" => wrapper["steps"][0]["params"] = serde_json::Value::Null,
            "nested-reserved-key" => {
                wrapper["steps"][0]["params"]["pipeline_steps_v1"] = json!([]);
            }
            _ => unreachable!(),
        }
        reseal_js_role_v2_candidate(&mut package, &mut wrapper, rebuild_plan);
        let bytes = serde_json::to_vec(&package).unwrap();
        let refusal =
            validate_owner_facet_bytes(&FacetContentType::new(FACET_INITIAL_REFIT, 1), &bytes)
                .unwrap_err()
                .to_string();
        assert!(refusal.contains(expected), "{case}: {refusal}");
    }
}

#[test]
fn js_v2_role_facets_keep_state_inventory_opaque_and_bounded() {
    let (_, wrapper) = js_role_v2_source();
    let states: Vec<Vec<u8>> = serde_json::from_value(wrapper["states"].clone()).unwrap();
    assert_eq!(
        states.iter().map(Vec::len).collect::<Vec<_>>(),
        [202, 308, 5329]
    );
    for (state, sha) in states.iter().zip([
        "3465c352ca03eb6260debef987d0fb5e1f2e4d9b52c44618acc12d4ff3411ff1",
        "2eba4a3e1257fa268d06ce15a18dec0d4ea6d515954c47d94bdfa3c2aefec2f1",
        "79c5fe6d80c6b722f7cb44f22d66edc274d3f70dbecf46c7010eb68b5a16f5b1",
    ]) {
        assert!(state.starts_with(b"N4ME"));
        assert_eq!(format!("{:x}", sha2::Sha256::digest(state)), sha);
    }
    let validated = proof(FACET_INITIAL_REFIT, 1, JS_ROLE_V2_INITIAL);
    let inspection = inspect_facet_model_states(&validated).unwrap();
    assert_eq!(inspection.len(), 1);
    assert!(inspection.iter().all(|state| {
        state.verdict.status == FacetSemanticStatus::Unverified && state.native_descriptor.is_none()
    }));
    for case in ["empty-buffer", "missing-buffer", "extra-buffer", "non-n4me"] {
        let (mut package, mut wrapper) = js_role_v2_source();
        let states = wrapper["states"].as_array_mut().unwrap();
        match case {
            "empty-buffer" => states[0] = json!([]),
            "missing-buffer" => {
                states.pop();
            }
            "extra-buffer" => states.push(states[0].clone()),
            "non-n4me" => states[0][0] = json!(0),
            _ => unreachable!(),
        }
        reseal_js_role_v2_candidate(&mut package, &mut wrapper, false);
        let refusal = validate_owner_facet_bytes(
            &FacetContentType::new(FACET_INITIAL_REFIT, 1),
            &serde_json::to_vec(&package).unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(
            refusal.contains("complete native recipe and state sequence"),
            "{case}: {refusal}"
        );
    }
}
