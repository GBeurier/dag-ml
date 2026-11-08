use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use dag_ml_core::*;

// Genuine native RolePipeline / distinct-target Package3 exports; provenance
// and Methods state sidecars live beside these unchanged owner fixtures.
const OUTCOME: &[u8] = include_bytes!("fixtures/artifact_facets/native_role/training_outcome.json");
const PACKAGE: &[u8] =
    include_bytes!("fixtures/artifact_facets/native_role/predictor_package.json");
const CHILD: &[u8] = include_bytes!("fixtures/artifact_facets/native_role/child_refit_v3.json");

struct Source {
    members: BTreeMap<String, Vec<u8>>,
    reads: RefCell<BTreeSet<String>>,
}

impl Source {
    fn new(export: &ArtifactFacetExport) -> Self {
        Self {
            members: export.members.clone(),
            reads: RefCell::default(),
        }
    }
}

impl FacetMemberSource for Source {
    fn read_member(&self, path: &str) -> FacetResult<Vec<u8>> {
        assert!(
            self.reads.borrow_mut().insert(path.into()),
            "read each member once"
        );
        self.members
            .get(path)
            .cloned()
            .ok_or_else(|| FacetRefusal::InvalidStructure {
                detail: format!("missing test member `{path}`"),
            })
    }
}

#[test]
fn genuine_role_predictor_and_standalone_child_preserve_exact_roots_and_requirements() {
    let predictor = build_native_predictor_facets(OUTCOME, PACKAGE).unwrap();
    let closure =
        validate_selected_native_closure(&predictor.graph, "model", &Source::new(&predictor))
            .unwrap();
    assert_eq!(closure.root().original_bytes(), PACKAGE);
    assert_eq!(closure.evidence().unwrap().original_bytes(), OUTCOME);
    let package = closure.root().predictor_package().unwrap();
    assert_eq!(
        closure.state_members().len(),
        package.execution_bundle.refit_artifacts.len()
    );
    for requirement in closure.methods_requirements() {
        assert_eq!(requirement.artifact_kind, "methods_role_pipeline");
        assert_eq!((requirement.abi_major, requirement.abi_min_minor), (2, 14));
    }
    assert!(!closure.methods_requirements().is_empty());
    for state in closure.state_members() {
        let record = package
            .execution_bundle
            .refit_artifacts
            .iter()
            .find(|record| record.artifact.id == state.artifact_id)
            .unwrap();
        assert_eq!(
            predictor.members[&state.member_path],
            package.execution_bundle.raw_artifact_payloads[&record.artifact.id]
        );
    }

    let child = build_native_refit_facets(CHILD).unwrap();
    assert_eq!(
        child
            .graph
            .artifacts
            .iter()
            .map(|artifact| artifact.id.as_str())
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["recipe", "full_refit", "refit_run"])
    );
    assert!(child.graph.links.iter().all(|link| !matches!(
        link.relation,
        ArtifactFacetRelation::RefitOf | ArtifactFacetRelation::Evaluates
    )));
    let closure =
        validate_selected_native_closure(&child.graph, "full_refit", &Source::new(&child)).unwrap();
    assert_eq!(closure.root().original_bytes(), CHILD);
    assert!(closure.evidence().is_none());
    assert!(!closure.state_members().is_empty());
    let package = closure.root().refit_package().unwrap();
    for state in closure.state_members() {
        let record = package
            .outcome
            .execution_bundle
            .refit_artifacts
            .iter()
            .find(|record| record.artifact.id == state.artifact_id)
            .unwrap();
        assert_eq!(
            child.members[&state.member_path],
            package.outcome.execution_bundle.raw_artifact_payloads[&record.artifact.id]
        );
    }
}

#[test]
fn selected_native_read_does_not_fetch_foreign_content_and_companion_bytes_may_reformat() {
    let mut export = build_native_predictor_facets(OUTCOME, PACKAGE).unwrap();
    export.graph.artifacts.push(ArtifactFacet {
        id: "foreign".into(),
        kind: ArtifactFacetKind::Attachment,
        owner: "other-owner".into(),
        content_type: FacetContentType::new("other.document", 1),
        identity: OwnerIdentity {
            owner: "other-owner".into(),
            schema: "other.document".into(),
            schema_version: 1,
            fingerprint_profile: "opaque-v1".into(),
            fingerprint: "foreign-identity".into(),
        },
        storage: ArtifactFacetStorage::Embedded {
            root_member: "foreign/opaque.bin".into(),
            member_paths: vec!["foreign/opaque.bin".into()],
        },
        requirements: None,
    });
    let mut source = Source::new(&export);
    // No foreign bytes are available: a read would fail. The real Core inventory
    // still owns integrity and existence; DAG never parses these opaque members.
    for id in ["bundle", "plan"] {
        let artifact = export
            .graph
            .artifacts
            .iter()
            .find(|artifact| artifact.id == id)
            .unwrap();
        let ArtifactFacetStorage::Embedded { root_member, .. } = &artifact.storage else {
            panic!("embedded companion")
        };
        let bytes = source.members.get_mut(root_member).unwrap();
        let mut formatted = b" \n".to_vec();
        formatted.extend_from_slice(bytes);
        formatted.extend_from_slice(b"\n ");
        *bytes = formatted;
    }
    validate_selected_native_closure(&export.graph, "model", &source).unwrap();
    assert!(!source.reads.borrow().contains("foreign/opaque.bin"));
}

#[test]
fn resigned_stale_bundle_and_plan_companions_refuse() {
    for id in ["bundle", "plan"] {
        let mut export = build_native_predictor_facets(OUTCOME, PACKAGE).unwrap();
        let mut source = Source::new(&export);
        let artifact = export
            .graph
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.id == id)
            .unwrap();
        let ArtifactFacetStorage::Embedded { root_member, .. } = &artifact.storage else {
            panic!("embedded companion")
        };
        let bytes = if id == "bundle" {
            let mut bundle = closure_package().execution_bundle;
            bundle.bundle_id = BundleId::new("bundle:stale-but-valid").unwrap();
            bundle.validate().unwrap();
            serde_json::to_vec(&bundle).unwrap()
        } else {
            let mut plan = closure_package().effective_plan;
            plan.id = "plan:stale-but-valid".into();
            plan.validate().unwrap();
            serde_json::to_vec(&plan).unwrap()
        };
        // Re-sign the descriptor as well: raw inventory/hash checks alone pass.
        let stale = validate_owner_facet_bytes(&artifact.content_type, &bytes)
            .unwrap()
            .unwrap();
        artifact.identity = stale.identity().clone();
        source.members.insert(root_member.clone(), bytes);
        assert!(matches!(
            validate_selected_native_closure(&export.graph, "model", &source),
            Err(FacetRefusal::Mismatch { .. })
        ));
    }
}

fn closure_package() -> PortablePredictorPackage {
    PortablePredictorPackage::from_json(std::str::from_utf8(PACKAGE).unwrap()).unwrap()
}

#[test]
fn host_sidecar_package_cannot_enter_native_write_or_read() {
    let outcome = TrainingOutcome::from_json(std::str::from_utf8(OUTCOME).unwrap()).unwrap();
    let package = outcome
        .to_portable_predictor_package(
            "package:host-sidecar",
            FittedArtifactMode::AllowHostSidecar,
            ArtifactLoadMode::HostSidecar,
        )
        .unwrap();
    let bytes = serde_json::to_vec(&package).unwrap();
    assert!(build_native_predictor_facets(OUTCOME, &bytes).is_err());
    let loose = build_training_artifact_facets_from_bytes(OUTCOME, &bytes).unwrap();
    assert!(validate_selected_native_closure(&loose.graph, "model", &Source::new(&loose)).is_err());
}

#[test]
fn injected_scientific_link_and_raw_state_tamper_refuse() {
    let mut export = build_native_predictor_facets(OUTCOME, PACKAGE).unwrap();
    export.graph.links.push(ArtifactFacetLink {
        id: "link:injected".into(),
        owner: "dag-ml".into(),
        relation: ArtifactFacetRelation::Evaluates,
        source_artifact: "score".into(),
        target_artifact: "evidence".into(),
    });
    assert!(matches!(
        validate_selected_native_closure(&export.graph, "model", &Source::new(&export)),
        Err(FacetRefusal::Mismatch { .. })
    ));
    export.graph.links.pop();
    let mut source = Source::new(&export);
    let state =
        expected_facet_state_members("model", &export.context.owner_proofs["model"]).unwrap();
    source.members.get_mut(&state[0].member_path).unwrap()[0] ^= 1;
    assert!(matches!(
        validate_selected_native_closure(&export.graph, "model", &source),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn fixed_native_root_collision_refuses_explicitly() {
    let mut export = build_native_predictor_facets(OUTCOME, PACKAGE).unwrap();
    let mut other = export
        .graph
        .artifacts
        .iter()
        .find(|artifact| artifact.id == "model")
        .unwrap()
        .clone();
    other.id = "another-native-model".into();
    other.storage = ArtifactFacetStorage::Reference { root_sha256: None };
    export.graph.artifacts.push(other);
    let error = validate_selected_native_closure(&export.graph, "model", &Source::new(&export))
        .unwrap_err();
    assert!(error.to_string().contains("collision refused"));
}
