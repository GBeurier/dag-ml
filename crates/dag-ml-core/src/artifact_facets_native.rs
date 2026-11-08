//! Strict native scientific closure, independent of physical containers.
use super::*;
use crate::{ArtifactBackend, FittedArtifactMode};
use std::collections::btree_map::Entry;

/// Core supplies verified member bytes and charges its own host memory budget.
/// This interface neither opens containers nor grants execution admission.
pub trait FacetMemberSource {
    fn read_member(&self, path: &str) -> FacetResult<Vec<u8>>;
}

/// A requirement derived from validated owner content, never a manifest hint.
/// N4MC's importer also makes its producer minor a runtime admission minimum;
/// other wire families retain their declared capability floors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeMethodsRequirement {
    pub artifact_id: crate::ArtifactId,
    pub artifact_kind: String,
    pub abi_major: u32,
    pub abi_min_minor: u32,
}

/// Process-local proof of the selected scientific closure. Actual verified
/// inventory, memory admission and Methods state inspection remain caller-owned.
#[derive(Clone, Debug)]
pub struct NativePackageClosure {
    root: ValidatedOwnerArtifact,
    evidence: Option<ValidatedOwnerArtifact>,
    state_members: Vec<FacetStateMember>,
    methods_requirements: Vec<NativeMethodsRequirement>,
}

impl NativePackageClosure {
    pub fn root(&self) -> &ValidatedOwnerArtifact {
        &self.root
    }
    pub fn evidence(&self) -> Option<&ValidatedOwnerArtifact> {
        self.evidence.as_ref()
    }
    pub fn state_members(&self) -> &[FacetStateMember] {
        &self.state_members
    }
    pub fn methods_requirements(&self) -> &[NativeMethodsRequirement] {
        &self.methods_requirements
    }
}

fn native_path(path: &str, extension: &str) -> bool {
    path.starts_with("methods/")
        && path.ends_with(extension)
        && !path.contains('\\')
        && path
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn native_requirements(
    records: &[crate::RefitArtifactRecord],
    raw_artifact_payloads: &BTreeMap<crate::ArtifactId, Vec<u8>>,
) -> FacetResult<Vec<NativeMethodsRequirement>> {
    let mut requirements = Vec::new();
    for record in records {
        let artifact = &record.artifact;
        let abi = match artifact.kind.as_str() {
            "n4m_model" => Some(crate::hpo::methods_n4mm_abi_requirement(artifact)?),
            crate::NATIVE_ESTIMATOR_ARTIFACT_KIND => Some((
                crate::hpo::METHODS_ABI_MAJOR,
                crate::hpo::METHODS_N4ME_MIN_ABI_MINOR,
            )),
            "methods_role_pipeline" => Some((crate::hpo::METHODS_ABI_MAJOR, 14)),
            "methods_role_classifier_pipeline" => Some((crate::hpo::METHODS_ABI_MAJOR, 17)),
            "methods_multimodal_classifier_pipeline" => {
                // The root proof already validated the wrapper, signed recipe,
                // raw identity and complete nested state. Read only its state
                // projection here; ArtifactRef ABI hints cannot override bytes.
                #[derive(serde::Deserialize)]
                struct N4mcState {
                    state: Vec<u8>,
                }
                let raw = raw_artifact_payloads
                    .get(&artifact.id)
                    .ok_or_else(|| invalid_payload("missing validated N4MC wrapper"))?;
                let wrapper: N4mcState = serde_json::from_slice(raw)?;
                Some(
                    crate::methods_classification_state::validated_n4mc_abi_requirement(
                        &wrapper.state,
                    )?,
                )
            }
            // N4MF has no producer-minor restriction; do not invent one.
            _ => None,
        };
        if let Some((abi_major, abi_min_minor)) = abi {
            requirements.push(NativeMethodsRequirement {
                artifact_id: artifact.id.clone(),
                artifact_kind: artifact.kind.clone(),
                abi_major,
                abi_min_minor,
            });
        }
    }
    Ok(requirements)
}

/// Validate the common raw/identity/coverage boundary and each supported family.
/// Opaque N4MM/N4ME state is inspected by Methods at execution admission.
fn native_records(
    plan: &ExecutionPlan,
    records: &[crate::RefitArtifactRecord],
    raw_artifact_payloads: &BTreeMap<crate::ArtifactId, Vec<u8>>,
    child: bool,
) -> FacetResult<()> {
    let mut ids = BTreeSet::new();
    let mut uris = BTreeSet::new();
    for record in records {
        let artifact = &record.artifact;
        artifact.validate_portable()?;
        if !ids.insert(&artifact.id) {
            return Err(mismatch("duplicate native refit artifact ID"));
        }
        let bytes = raw_artifact_payloads
            .get(&artifact.id)
            .ok_or_else(|| invalid_payload("missing native raw artifact payload"))?;
        let uri = artifact
            .uri
            .as_deref()
            .ok_or_else(|| invalid_payload("native artifact requires a portable URI"))?;
        if !uris.insert(uri) {
            return Err(mismatch("native artifact URIs must be unique"));
        }
        if artifact.backend != Some(ArtifactBackend::Raw)
            || artifact.size_bytes != Some(bytes.len() as u64)
            || artifact.content_fingerprint.as_deref() != Some(sha256(bytes).as_str())
        {
            return Err(mismatch("native raw payload, size or fingerprint mismatch"));
        }
        match artifact.kind.as_str() {
            "n4m_model" => {
                if artifact.plugin.is_some()
                    || artifact.plugin_version.is_some()
                    || !native_path(uri, ".n4mm")
                {
                    return Err(invalid_payload(
                        "N4MM requires a plugin-free raw artifact and safe methods/*.n4mm URI",
                    ));
                }
                crate::hpo::methods_n4mm_abi_requirement(artifact)?;
                if artifact
                    .native_predictor_descriptor
                    .as_ref()
                    .is_some_and(|descriptor| {
                        descriptor.format_version == 2 && descriptor.pipeline.is_none()
                    })
                {
                    return Err(invalid_payload(
                        "N4MM format 2 requires an embedded pipeline descriptor",
                    ));
                }
            }
            crate::NATIVE_ESTIMATOR_ARTIFACT_KIND if child => {
                if artifact.plugin.is_some()
                    || artifact.plugin_version.is_some()
                    || artifact.abi_major != Some(crate::hpo::METHODS_ABI_MAJOR)
                    || artifact.abi_min_minor != Some(crate::hpo::METHODS_N4ME_MIN_ABI_MINOR)
                    || !native_path(uri, ".n4me")
                    || !bytes.starts_with(b"N4ME")
                {
                    return Err(invalid_payload("N4ME requires plugin-free raw state, safe methods/*.n4me URI and exact Methods ABI 2.13"));
                }
            }
            "methods_role_pipeline" => {
                role_payload::validate_role_payload(record, bytes)?;
                role_payload::validate_methods_role_pipeline_recipe(record, bytes, plan)?;
                if child
                    && artifact.controller_id.as_str()
                        != crate::METHODS_NATIVE_REGRESSION_CONTROLLER
                {
                    return Err(invalid_payload(
                        "Package3 role pipelines require the native Methods controller",
                    ));
                }
            }
            "methods_multimodal_pipeline" if !child => {
                crate::validate_methods_multimodal_pipeline_payload(record, bytes)?;
                crate::validate_methods_multimodal_pipeline_recipe(record, bytes, plan)?;
                // No minimum ABI is declared for this historical wrapper family.
            }
            "methods_role_classifier_pipeline" | "methods_multimodal_classifier_pipeline"
                if !child =>
            {
                crate::validate_methods_classifier_payload(record, bytes, plan)?;
            }
            _ => {
                return Err(unsupported_payload(format!(
                    "unsupported native Package{} artifact family `{}`",
                    if child { 3 } else { 2 },
                    artifact.kind
                )))
            }
        }
    }
    if ids.is_empty() || ids != raw_artifact_payloads.keys().collect::<BTreeSet<_>>() {
        return Err(mismatch(
            "native raw states must exactly cover all refit artifacts",
        ));
    }
    Ok(())
}

fn native_predictor(
    outcome: &TrainingOutcome,
    package: &PortablePredictorPackage,
) -> FacetResult<()> {
    // Both values come from current owner proofs. build_training_proofs below
    // additionally enforces their exact cross-links before publishing facets.
    if package.schema_version != 2 || outcome.schema_version != 2 {
        return Err(invalid_payload(
            "native predictor requires Package2 and TrainingOutcome2",
        ));
    }
    if package.fitted_artifact_mode != FittedArtifactMode::PortableRequired
        || package
            .artifact_bindings
            .iter()
            .any(|binding| binding.load_mode != ArtifactLoadMode::NativePortable)
    {
        return Err(invalid_payload(
            "native predictor refuses host-sidecar artifacts",
        ));
    }
    if let Some(caches) = &outcome.portable_prediction_caches {
        caches.validate_against_bundle(&outcome.execution_bundle)?;
        if caches.schema_version != 2 {
            return Err(invalid_payload("native predictor requires cache schema2"));
        }
    } else {
        let bundle = &outcome.execution_bundle;
        if !bundle.prediction_requirements.is_empty()
            || !bundle.prediction_caches.is_empty()
            || outcome
                .effective_plan
                .graph_plan
                .graph
                .edges
                .iter()
                .any(|edge| edge.contract.requires_oof)
        {
            return Err(invalid_payload(
                "absent prediction caches require proof of no OOF cache dependency",
            ));
        }
    }
    if outcome.score_set.schema_version != 2 {
        return Err(invalid_payload("native predictor requires score schema2"));
    }
    native_records(
        &package.effective_plan,
        &package.execution_bundle.refit_artifacts,
        &package.execution_bundle.raw_artifact_payloads,
        false,
    )?;
    let records = package
        .execution_bundle
        .refit_artifacts
        .iter()
        .map(|record| &record.artifact.id)
        .collect::<BTreeSet<_>>();
    let bindings = package
        .artifact_bindings
        .iter()
        .map(|binding| &binding.artifact_id)
        .collect::<BTreeSet<_>>();
    if records != bindings || bindings.len() != package.artifact_bindings.len() {
        return Err(mismatch(
            "native bindings must exactly cover all refit artifacts",
        ));
    }
    Ok(())
}

fn native_refit(package: &PortableRefitPackageV3) -> FacetResult<()> {
    // The caller obtained this package through native_proof, including the
    // owner validator and all declarative composite checks.
    if package.schema_version != 3 || package.outcome.schema_version != 3 {
        return Err(invalid_payload(
            "native refit requires exact Package3 and outcome3",
        ));
    }
    native_records(
        &package.outcome.effective_plan,
        &package.outcome.execution_bundle.refit_artifacts,
        &package.outcome.execution_bundle.raw_artifact_payloads,
        true,
    )
}

/// Retain exact supplied roots; derive host-stable companions. The same strict
/// scientific gate is rerun when reading the selected closure.
pub fn build_native_predictor_facets(
    outcome_bytes: &[u8],
    package_bytes: &[u8],
) -> FacetResult<ArtifactFacetExport> {
    let outcome = native_proof(FACET_TRAINING_OUTCOME, 2, outcome_bytes)?;
    let package = native_proof(FACET_PREDICTOR_PACKAGE, 2, package_bytes)?;
    native_predictor(
        outcome
            .training_outcome()
            .ok_or_else(|| error("expected training outcome"))?,
        package
            .predictor_package()
            .ok_or_else(|| error("expected predictor package"))?,
    )?;
    build_training_proofs(outcome, package)
}

/// Standalone target-bound child. No parent, source score or RefitOf is invented.
pub fn build_native_refit_facets(package_bytes: &[u8]) -> FacetResult<ArtifactFacetExport> {
    let proof = native_proof(FACET_REFIT_V3, 3, package_bytes)?;
    let package = proof
        .refit_package()
        .ok_or_else(|| error("expected refit package"))?;
    native_refit(package)?;
    let mut export = ArtifactFacetExport::new();
    export.add(
        "recipe",
        ArtifactFacetKind::Recipe,
        owner_object(
            FACET_GRAPH_SPEC,
            1,
            &package.outcome.effective_plan.graph_plan.graph,
        )?,
        Some("dagml/recipe.json"),
    )?;
    export.add(
        "full_refit",
        ArtifactFacetKind::FullRefit,
        proof.clone(),
        Some("dagml/full_refit.json"),
    )?;
    export.add("refit_run", ArtifactFacetKind::Run, proof.clone(), None)?;
    export.add_raw_members(
        "full_refit",
        &package.outcome.execution_bundle.raw_artifact_payloads,
    )?;
    export.link(ArtifactFacetRelation::RecipeOf, "recipe", "full_refit");
    export.link(ArtifactFacetRelation::RecipeOf, "recipe", "refit_run");
    export.link(ArtifactFacetRelation::ProducedBy, "full_refit", "refit_run");
    export.validate()?;
    Ok(export)
}

fn native_proof(id: &str, version: u32, bytes: &[u8]) -> FacetResult<ValidatedOwnerArtifact> {
    validate_owner_facet_bytes(&FacetContentType::new(id, version), bytes)?
        .ok_or_else(|| unsupported_payload(id))
}

fn embedded_root<'a>(graph: &'a ArtifactFacetGraph, id: &str) -> FacetResult<&'a str> {
    let artifact = graph
        .artifacts
        .iter()
        .find(|artifact| artifact.id == id)
        .ok_or_else(|| error(format!("selected native closure lacks `{id}`")))?;
    match &artifact.storage {
        ArtifactFacetStorage::Embedded { root_member, .. } => Ok(root_member),
        _ => Err(error("native root and evidence must be embedded")),
    }
}

/// Reconstruct the exact expected scientific graph from selected root/evidence
/// bytes. Compare every companion identity and every DAG link touching it.
/// Unrelated content is never read, interpreted or used as evidence.
pub fn validate_selected_native_closure(
    graph: &ArtifactFacetGraph,
    root_id: &str,
    source: &dyn FacetMemberSource,
) -> FacetResult<NativePackageClosure> {
    graph.validate_structure()?;
    if !matches!(root_id, "model" | "full_refit") {
        return Err(unsupported_payload(
            "native profile requires fixed root model or full_refit",
        ));
    }
    let root_path = embedded_root(graph, root_id)?;
    let root_bytes = source.read_member(root_path)?;
    let mut members = BTreeMap::from([(root_path.to_owned(), root_bytes)]);
    let expected = if root_id == "model" {
        let outcome_path = embedded_root(graph, "result")?;
        if let Entry::Vacant(entry) = members.entry(outcome_path.to_owned()) {
            entry.insert(source.read_member(outcome_path)?);
        }
        build_native_predictor_facets(&members[outcome_path], &members[root_path])?
    } else {
        build_native_refit_facets(&members[root_path])?
    };
    let selected = expected
        .graph
        .artifacts
        .iter()
        .map(|artifact| artifact.id.as_str())
        .collect::<BTreeSet<_>>();
    for artifact in &graph.artifacts {
        if !selected.contains(artifact.id.as_str())
            && artifact.owner == "dag-ml"
            && matches!(
                artifact.content_type.id.as_str(),
                FACET_PREDICTOR_PACKAGE | FACET_REFIT_V3
            )
        {
            return Err(mismatch(
                "fixed native facet IDs permit only one native closure; collision refused",
            ));
        }
    }
    for artifact in &expected.graph.artifacts {
        let actual = graph
            .artifacts
            .iter()
            .find(|candidate| candidate.id == artifact.id)
            .ok_or_else(|| mismatch(format!("missing scientific companion `{}`", artifact.id)))?;
        if actual != artifact {
            return Err(mismatch(format!("native scientific facet `{}` differs from reconstructed closure (fixed-ID collision or stale evidence)", artifact.id)));
        }
        if let ArtifactFacetStorage::Embedded {
            root_member,
            member_paths,
        } = &actual.storage
        {
            for path in member_paths {
                if let Entry::Vacant(entry) = members.entry(path.clone()) {
                    entry.insert(source.read_member(path)?);
                }
            }
            let proof = &expected.context.owner_proofs[&actual.id];
            let actual_proof;
            let proof = if matches!(actual.id.as_str(), "model" | "full_refit" | "result") {
                if proof.original_bytes() != members[root_member] {
                    return Err(mismatch("native root/evidence exact bytes changed"));
                }
                proof
            } else {
                actual_proof =
                    validate_owner_facet_bytes(&actual.content_type, &members[root_member])?
                        .ok_or_else(|| unsupported_payload(actual.content_type.id.clone()))?;
                &actual_proof
            };
            matches_proof(actual, proof)?;
            // Root/evidence bytes are exactly those supplied to the rebuilding
            // gate. Derived companions compare by owner identity, not JSON order.
            validate_facet_members(actual, proof, &members)?;
        }
    }
    for link in &expected.graph.links {
        if !graph.links.contains(link) {
            return Err(mismatch(
                "selected native closure lacks a reconstructed scientific link",
            ));
        }
    }
    for link in &graph.links {
        if link.owner == "dag-ml"
            && (selected.contains(link.source_artifact.as_str())
                || selected.contains(link.target_artifact.as_str()))
            && !expected.graph.links.contains(link)
        {
            return Err(mismatch(
                "injected DAG link touches the selected native scientific closure",
            ));
        }
    }
    // Matching evaluation links remain diagnostic; strict native_predictor above
    // established the actual score/outcome binding, independently of link status.
    // The rebuilt export already validated its expected links and members.
    // Transfer its proofs without copying the root's retained raw state again.
    let mut proofs = expected.context.owner_proofs;
    let root = proofs
        .remove(root_id)
        .ok_or_else(|| error("missing rebuilt root proof"))?;
    let evidence = proofs.remove("evidence");
    let (_, records, raw_artifact_payloads) =
        model_closure(&root.payload).ok_or_else(|| error("expected native package"))?;
    let methods_requirements = native_requirements(records, raw_artifact_payloads)?;
    let state_members = expected_facet_state_members(root_id, &root)?;
    Ok(NativePackageClosure {
        root,
        evidence,
        state_members,
        methods_requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn n4mc_requirements_use_validated_producer_bytes_not_artifact_abi_hints() {
        // Header projection witness only. The private caller runs after the
        // complete owner proof; this unit creates no scientific closure proof.
        let mut state = b"N4MC".to_vec();
        for value in [1_u32, 2, 20, 0] {
            state.extend(value.to_le_bytes());
        }
        let raw = serde_json::to_vec(&json!({"state": state})).unwrap();
        let digest = sha256(&raw);
        let record: crate::RefitArtifactRecord = serde_json::from_value(json!({
            "node_id":"model:n4mc", "controller_id":"controller:methods.python.multimodal.classification",
            "params_fingerprint":"a".repeat(64),
            "artifact":{"id":"artifact:n4mc", "kind":"methods_multimodal_classifier_pipeline",
                "controller_id":"controller:methods.python.multimodal.classification", "backend":"raw",
                "uri":format!("artifacts/{digest}.json"), "content_fingerprint":digest,
                "size_bytes":raw.len(), "abi_major":2, "abi_min_minor":17}
        })).unwrap();
        let raw_payloads = BTreeMap::from([(record.artifact.id.clone(), raw)]);
        let requirements =
            native_requirements(std::slice::from_ref(&record), &raw_payloads).unwrap();
        assert_eq!(
            requirements,
            vec![NativeMethodsRequirement {
                artifact_id: record.artifact.id.clone(),
                artifact_kind: record.artifact.kind.clone(),
                abi_major: 2,
                abi_min_minor: 20,
            }]
        );
        let required = &requirements[0];
        assert!(crate::hpo::validate_methods_abi_compatibility(
            2,
            17,
            required.abi_major,
            required.abi_min_minor
        )
        .is_err());
        crate::hpo::validate_methods_abi_compatibility(
            2,
            20,
            required.abi_major,
            required.abi_min_minor,
        )
        .unwrap();

        // Other wire importers do not require runtime minor >= producer minor.
        let mut other = record;
        for (kind, expected_minor) in [
            (crate::NATIVE_ESTIMATOR_ARTIFACT_KIND, Some(13)),
            ("methods_role_classifier_pipeline", Some(17)),
            ("methods_multimodal_pipeline", None),
        ] {
            other.artifact.kind = kind.into();
            let requirements =
                native_requirements(std::slice::from_ref(&other), &raw_payloads).unwrap();
            assert_eq!(
                requirements.first().map(|required| required.abi_min_minor),
                expected_minor
            );
        }
        other.artifact.kind = "n4m_model".into();
        other.artifact.controller_id =
            crate::ControllerId::new(crate::METHODS_PLS_CONTROLLER_ID).unwrap();
        other.controller_id = other.artifact.controller_id.clone();
        other.artifact.abi_min_minor = Some(crate::hpo::METHODS_PLS_N4MM_MIN_ABI_MINOR);
        let requirements =
            native_requirements(std::slice::from_ref(&other), &raw_payloads).unwrap();
        assert_eq!(
            requirements[0].abi_min_minor,
            crate::hpo::METHODS_PLS_N4MM_MIN_ABI_MINOR
        );
    }
}

#[cfg(test)]
mod legacy_semantics_tests {
    use super::*;
    use crate::bundle::{BundlePredictionCacheRecord, BundlePredictionRequirement};
    use crate::bundle::{ExecutionBundle, PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION};
    use crate::canonical::parse_typed_json;
    use serde_json::{json, Value};
    use std::collections::BTreeSet;

    fn empty_bundle() -> ExecutionBundle {
        serde_json::from_value(json!({
            "bundle_id": "bundle:native.empty-cache", "schema_version": 2,
            "plan_id": "plan:native.empty-cache", "graph_fingerprint": "a".repeat(64),
            "campaign_fingerprint": "b".repeat(64), "controller_fingerprint": "c".repeat(64),
        }))
        .unwrap()
    }

    fn cache_free_graph() -> GraphSpec {
        serde_json::from_str(include_str!("../tests/fixtures/package/minimal_graph.json")).unwrap()
    }

    fn predictor_facets(
        outcome: &TrainingOutcome,
        package: &PortablePredictorPackage,
    ) -> FacetResult<ArtifactFacetExport> {
        build_native_predictor_facets(
            &crate::stable_json::to_vec(outcome).unwrap(),
            &crate::stable_json::to_vec(package).unwrap(),
        )
    }

    struct Members(BTreeMap<String, Vec<u8>>);
    impl FacetMemberSource for Members {
        fn read_member(&self, path: &str) -> FacetResult<Vec<u8>> {
            self.0
                .get(path)
                .cloned()
                .ok_or_else(|| invalid_payload(format!("missing {path}")))
        }
    }

    fn selected(export: &ArtifactFacetExport) -> FacetResult<NativePackageClosure> {
        validate_selected_native_closure(&export.graph, "model", &Members(export.members.clone()))
    }

    fn resign_outcome(outcome: &mut TrainingOutcome) {
        let plan_json = serde_json::to_string(&outcome.effective_plan).unwrap();
        outcome.effective_plan_fingerprint =
            parse_typed_json(&plan_json).unwrap().fingerprint().unwrap();
        outcome.outcome_fingerprint = "0".repeat(64);
        outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
    }

    // These are opaque transport witnesses, not numerical N4ME qualification.
    // The real five-model Methods campaign runs in the Node/web smoke script.
    fn role_transport_fixture(mixed: bool) -> (TrainingOutcome, PortablePredictorPackage) {
        let mut outcome = TrainingOutcome::from_json(include_str!(
            "../tests/fixtures/package/archive/training_outcome_port_explicit.json"
        ))
        .unwrap();
        outcome.execution_bundle.raw_artifact_payloads.clear();
        for (index, record) in outcome
            .execution_bundle
            .refit_artifacts
            .iter_mut()
            .enumerate()
        {
            let role = !mixed || index % 2 == 0;
            let payload = if role {
                serde_json::to_vec(&json!({
                    "schema": "dagml.methods.regression.v1", "node_id": record.node_id,
                    "params_fingerprint": record.params_fingerprint,
                    "target_names": ["y"], "feature_names": ["source:feature"],
                    "steps": [{"class": "n4m:models.regularized.ridge",
                        "params": outcome.effective_plan.node_plans[&record.node_id].params}],
                    "states": [[78, 52, 77, 69, index as u8]]
                }))
                .unwrap()
            } else {
                format!("opaque N4MM transport witness {index}").into_bytes()
            };
            let fingerprint = sha256(&payload);
            record.artifact.kind = if role {
                "methods_role_pipeline"
            } else {
                "n4m_model"
            }
            .into();
            record.artifact.backend = Some(ArtifactBackend::Raw);
            record.artifact.uri = Some(if role {
                format!("artifacts/{fingerprint}.json")
            } else {
                format!("methods/mixed-{index}.n4mm")
            });
            record.artifact.content_fingerprint = Some(fingerprint);
            record.artifact.size_bytes = Some(payload.len() as u64);
            record.artifact.plugin = role.then(|| "dagml.methods.wasm.regression".into());
            record.artifact.plugin_version = role.then(|| "1.0.0".into());
            record.artifact.abi_major = (!role).then_some(crate::hpo::METHODS_ABI_MAJOR);
            record.artifact.abi_min_minor =
                (!role).then_some(crate::hpo::METHODS_PLS_N4MM_MIN_ABI_MINOR);
            record.artifact.native_predictor_descriptor = None;
            record.artifact.native_estimator_descriptor = None;
            outcome
                .execution_bundle
                .raw_artifact_payloads
                .insert(record.artifact.id.clone(), payload);
        }
        let role_nodes = outcome
            .execution_bundle
            .refit_artifacts
            .iter()
            .filter(|record| record.artifact.kind == "methods_role_pipeline")
            .map(|record| record.node_id.clone())
            .collect::<BTreeSet<_>>();
        for node in &mut outcome.effective_plan.graph_plan.graph.nodes {
            if role_nodes.contains(&node.id) {
                node.operator = Some(json!({"type": "n4m:models.regularized.ridge"}));
            }
        }
        outcome.effective_plan.graph_fingerprint =
            crate::campaign::stable_json_fingerprint(&outcome.effective_plan.graph_plan.graph)
                .unwrap();
        outcome.execution_bundle.graph_fingerprint =
            outcome.effective_plan.graph_fingerprint.clone();
        let artifacts = outcome
            .execution_bundle
            .refit_artifacts
            .iter()
            .map(|record| (record.artifact.id.clone(), record.artifact.clone()))
            .collect::<BTreeMap<_, _>>();
        for record in &mut outcome.lineage {
            for artifact in &mut record.artifact_refs {
                if let Some(portable) = artifacts.get(&artifact.id) {
                    *artifact = portable.clone();
                }
            }
        }
        resign_outcome(&mut outcome);
        outcome.validate().unwrap();
        let package = outcome
            .to_portable_predictor_package(
                "predictor:archive.roles",
                FittedArtifactMode::PortableRequired,
                ArtifactLoadMode::NativePortable,
            )
            .unwrap();
        (outcome, package)
    }

    #[test]
    fn mixed_classifier_and_regression_requirements_retain_exact_families() {
        // Requirement projection only, like the old storage inventory witness.
        // Full fitted classifier admission remains covered by its owner tests.
        let (_, package) = role_transport_fixture(true);
        let mut records = package.execution_bundle.refit_artifacts.clone();
        let mut raw = package.execution_bundle.raw_artifact_payloads.clone();
        for (kind, suffix) in [
            ("methods_role_classifier_pipeline", "role-classifier"),
            (
                "methods_multimodal_classifier_pipeline",
                "multimodal-classifier",
            ),
            ("methods_multimodal_pipeline", "multimodal-regression"),
        ] {
            let mut record = records[0].clone();
            record.artifact.id = crate::ArtifactId::new(format!("artifact:{suffix}")).unwrap();
            record.artifact.kind = kind.into();
            let mut state = b"N4MC".to_vec();
            for value in [1_u32, 2, 20, 0] {
                state.extend(value.to_le_bytes());
            }
            raw.insert(
                record.artifact.id.clone(),
                serde_json::to_vec(&json!({"state": state})).unwrap(),
            );
            records.push(record);
        }
        let requirements = native_requirements(&records, &raw).unwrap();
        for record in &records {
            let requirement = requirements
                .iter()
                .find(|entry| entry.artifact_id == record.artifact.id);
            let minor = match record.artifact.kind.as_str() {
                "n4m_model" => Some(crate::hpo::METHODS_PLS_N4MM_MIN_ABI_MINOR),
                "methods_role_pipeline" => Some(14),
                "methods_role_classifier_pipeline" => Some(17),
                "methods_multimodal_classifier_pipeline" => Some(20),
                "methods_multimodal_pipeline" => None,
                _ => unreachable!(),
            };
            assert_eq!(requirement.map(|entry| entry.abi_min_minor), minor);
            if let Some(requirement) = requirement {
                assert_eq!(requirement.artifact_kind, record.artifact.kind);
                assert_eq!(requirement.abi_major, 2);
            }
        }
    }

    #[test]
    fn native_facets_are_independent_of_nested_json_key_order() {
        fn reversed_json(value: &Value) -> String {
            match value {
                Value::Object(map) => format!(
                    "{{{}}}",
                    map.iter()
                        .rev()
                        .map(|(key, value)| format!(
                            "{}:{}",
                            serde_json::to_string(key).unwrap(),
                            reversed_json(value)
                        ))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                Value::Array(values) => format!(
                    "[{}]",
                    values
                        .iter()
                        .map(reversed_json)
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                _ => serde_json::to_string(value).unwrap(),
            }
        }

        let (outcome, package) = role_transport_fixture(false);
        let reordered_outcome =
            TrainingOutcome::from_json(&reversed_json(&serde_json::to_value(&outcome).unwrap()))
                .unwrap();
        let reordered_package = PortablePredictorPackage::from_json(&reversed_json(
            &serde_json::to_value(&package).unwrap(),
        ))
        .unwrap();
        assert_eq!(outcome, reordered_outcome);
        assert_eq!(package, reordered_package);
        let original = predictor_facets(&outcome, &package).unwrap();
        let rebuilt = predictor_facets(&reordered_outcome, &reordered_package).unwrap();
        assert_eq!(original.members, rebuilt.members);
        assert_eq!(original.graph, rebuilt.graph);
        selected(&rebuilt).unwrap();
    }

    #[test]
    fn role_and_mixed_facets_preserve_exact_package_and_raw_closure() {
        for mixed in [false, true] {
            let (outcome, package) = role_transport_fixture(mixed);
            let export = predictor_facets(&outcome, &package).unwrap();
            let closure = selected(&export).unwrap();
            assert_eq!(
                closure.root().original_bytes(),
                crate::stable_json::to_vec(&package).unwrap()
            );
            assert_eq!(
                closure.evidence().unwrap().original_bytes(),
                crate::stable_json::to_vec(&outcome).unwrap()
            );
            assert_eq!(
                closure.state_members().len(),
                package.artifact_bindings.len()
            );
            assert!(closure
                .methods_requirements()
                .iter()
                .any(|item| item.artifact_kind == "methods_role_pipeline"));
            assert_eq!(
                closure
                    .methods_requirements()
                    .iter()
                    .any(|item| item.artifact_kind == "n4m_model"),
                mixed
            );
            for state in closure.state_members() {
                let bytes = &export.members[&state.member_path];
                assert_eq!(
                    bytes,
                    &package.execution_bundle.raw_artifact_payloads[&state.artifact_id]
                );
                assert_eq!(state.sha256, sha256(bytes));
                assert_eq!(state.byte_length, bytes.len() as u64);
            }
            let mut missing = Members(export.members.clone());
            missing.0.remove(&closure.state_members()[0].member_path);
            assert!(validate_selected_native_closure(&export.graph, "model", &missing).is_err());
            let mut altered = Members(export.members.clone());
            let mut graph: Value = serde_json::from_slice(&altered.0["dagml/recipe.json"]).unwrap();
            graph["id"] = json!("graph:foreign");
            altered.0.insert(
                "dagml/recipe.json".into(),
                serde_json::to_vec(&graph).unwrap(),
            );
            assert!(validate_selected_native_closure(&export.graph, "model", &altered).is_err());
            let mut injected = predictor_facets(&outcome, &package).unwrap();
            injected
                .members
                .insert("unexpected.json".into(), b"{}".to_vec());
            assert!(injected.validate().is_err());
            let mut alias = export.graph.clone();
            let model = alias
                .artifacts
                .iter_mut()
                .find(|facet| facet.id == "model")
                .unwrap();
            if let ArtifactFacetStorage::Embedded { member_paths, .. } = &mut model.storage {
                member_paths.push("unbound.bin".into());
            }
            assert!(
                validate_selected_native_closure(&alias, "model", &Members(export.members))
                    .is_err()
            );
        }
    }

    #[test]
    fn role_wrapper_refuses_unknown_schema_identity_recipe_and_state() {
        let (_, package) = role_transport_fixture(false);
        let original = &package.execution_bundle.refit_artifacts[0];
        let bytes = &package.execution_bundle.raw_artifact_payloads[&original.artifact.id];
        role_payload::validate_role_payload(original, bytes).unwrap();
        let mut r_native = original.clone();
        r_native.artifact.plugin = Some("dagml.methods.r.regression".into());
        role_payload::validate_role_payload(&r_native, bytes).unwrap();
        r_native.artifact.plugin = Some("dagml.methods.unknown.regression".into());
        assert!(role_payload::validate_role_payload(&r_native, bytes).is_err());
        let mut untrusted = original.clone();
        untrusted.artifact.plugin_version = Some("999.0.0".into());
        assert!(role_payload::validate_role_payload(&untrusted, bytes).is_err());
        let mut external = original.clone();
        external.artifact.uri = Some("../external.json".into());
        assert!(role_payload::validate_role_payload(&external, bytes).is_err());
        for (pointer, value) in [
            ("/schema", json!("dagml.methods.regression.v999")),
            ("/node_id", json!("model:another")),
            ("/params_fingerprint", json!("f".repeat(64))),
            ("/steps/0/class", json!("sklearn.Ridge")),
            ("/states/0", json!([80, 73, 67, 75, 76, 69])),
            ("/target_names", json!([])),
            ("/feature_names", json!(["duplicate", "duplicate"])),
        ] {
            let mut wrapper: Value = serde_json::from_slice(bytes).unwrap();
            if pointer == "/steps/0/class" {
                wrapper["steps"][0]["class"] = value;
            } else {
                *wrapper.pointer_mut(pointer).unwrap() = value;
            }
            let modified = serde_json::to_vec(&wrapper).unwrap();
            let mut record = original.clone();
            let raw = sha256(&modified);
            record.artifact.uri = Some(format!("artifacts/{raw}.json"));
            record.artifact.content_fingerprint = Some(raw);
            record.artifact.size_bytes = Some(modified.len() as u64);
            assert!(
                role_payload::validate_role_payload(&record, &modified).is_err(),
                "{pointer}"
            );
        }
        let mut wrapper: Value = serde_json::from_slice(bytes).unwrap();
        wrapper["unexpected"] = json!(true);
        let modified = serde_json::to_vec(&wrapper).unwrap();
        let mut record = original.clone();
        let raw = sha256(&modified);
        record.artifact.uri = Some(format!("artifacts/{raw}.json"));
        record.artifact.content_fingerprint = Some(raw);
        record.artifact.size_bytes = Some(modified.len() as u64);
        assert!(role_payload::validate_role_payload(&record, &modified).is_err());
    }

    #[test]
    fn octave_role_plugin_requires_both_exact_controller_owners() {
        let (_, package) = role_transport_fixture(false);
        let original = &package.execution_bundle.refit_artifacts[0];
        let bytes = &package.execution_bundle.raw_artifact_payloads[&original.artifact.id];
        let mut octave = original.clone();
        octave.artifact.plugin = Some("dagml.methods.octave.regression".into());
        assert!(role_payload::validate_role_payload(&octave, bytes).is_err());
        octave.artifact.controller_id =
            crate::ControllerId::new("controller:methods.octave.regression").unwrap();
        assert!(role_payload::validate_role_payload(&octave, bytes).is_err());
        octave.controller_id = octave.artifact.controller_id.clone();
        role_payload::validate_role_payload(&octave, bytes).unwrap();
        let mut foreign_artifact = octave.clone();
        foreign_artifact.artifact.controller_id = original.artifact.controller_id.clone();
        assert!(role_payload::validate_role_payload(&foreign_artifact, bytes).is_err());
        for plugin in [
            "dagml.methods.octave.regression.v2",
            "dagml.methods.octave.unknown",
        ] {
            let mut unknown = octave.clone();
            unknown.artifact.plugin = Some(plugin.into());
            assert!(role_payload::validate_role_payload(&unknown, bytes).is_err());
        }
        let mut wrong_version = octave;
        wrong_version.artifact.plugin_version = Some("1.0.1".into());
        assert!(role_payload::validate_role_payload(&wrong_version, bytes).is_err());
    }

    #[test]
    fn python_role_plugin_requires_both_exact_controller_owners() {
        let (_, package) = role_transport_fixture(false);
        let original = &package.execution_bundle.refit_artifacts[0];
        let bytes = &package.execution_bundle.raw_artifact_payloads[&original.artifact.id];
        let mut python = original.clone();
        python.artifact.plugin = Some("dagml.methods.python.regression".into());
        assert!(role_payload::validate_role_payload(&python, bytes).is_err());
        python.artifact.controller_id =
            crate::ControllerId::new("controller:methods.python.regression").unwrap();
        assert!(role_payload::validate_role_payload(&python, bytes).is_err());
        python.controller_id = python.artifact.controller_id.clone();
        role_payload::validate_role_payload(&python, bytes).unwrap();
        let mut foreign = python.clone();
        foreign.artifact.controller_id = original.artifact.controller_id.clone();
        assert!(role_payload::validate_role_payload(&foreign, bytes).is_err());
        let mut unknown = python.clone();
        unknown.artifact.plugin = Some("dagml.methods.python.regression.v2".into());
        assert!(role_payload::validate_role_payload(&unknown, bytes).is_err());
        let mut wrong_version = python;
        wrong_version.artifact.plugin_version = Some("1.0.1".into());
        assert!(role_payload::validate_role_payload(&wrong_version, bytes).is_err());
    }

    #[test]
    fn resealed_role_recipe_cannot_override_the_unchanged_effective_plan() {
        for change_method in [false, true] {
            let (mut outcome, _) = role_transport_fixture(false);
            let record = &mut outcome.execution_bundle.refit_artifacts[0];
            let mut wrapper: Value = serde_json::from_slice(
                &outcome.execution_bundle.raw_artifact_payloads[&record.artifact.id],
            )
            .unwrap();
            if change_method {
                wrapper["steps"][0]["class"] = json!("n4m:models.pls");
            } else {
                wrapper["steps"][0]["params"]["unplanned_alpha"] = json!(9);
            }
            let bytes = serde_json::to_vec(&wrapper).unwrap();
            let raw = sha256(&bytes);
            record.artifact.uri = Some(format!("artifacts/{raw}.json"));
            record.artifact.content_fingerprint = Some(raw);
            record.artifact.size_bytes = Some(bytes.len() as u64);
            role_payload::validate_role_payload(record, &bytes).unwrap();
            let changed = record.artifact.clone();
            outcome
                .execution_bundle
                .raw_artifact_payloads
                .insert(changed.id.clone(), bytes);
            for lineage in &mut outcome.lineage {
                for artifact in &mut lineage.artifact_refs {
                    if artifact.id == changed.id {
                        *artifact = changed.clone();
                    }
                }
            }
            resign_outcome(&mut outcome);
            outcome.validate().unwrap();
            let package = outcome
                .to_portable_predictor_package(
                    "predictor:resealed.recipe",
                    FittedArtifactMode::PortableRequired,
                    ArtifactLoadMode::NativePortable,
                )
                .unwrap();
            assert!(predictor_facets(&outcome, &package)
                .unwrap_err()
                .to_string()
                .contains("saved recipe differs"));
        }
    }

    #[cfg(dag_ml_workspace_contract_fixtures)]
    #[test]
    fn package_local_archive_fixture_tracks_canonical_v2_replay_fixture() {
        // The package gate compiles without workspace fixtures.  This workspace
        // witness prevents the required package-local copy from silently
        // diverging from the canonical V2 replay fixture.
        assert_eq!(
            include_str!("../tests/fixtures/package/archive/training_outcome_port_explicit.json"),
            include_str!(
                "../../../examples/fixtures/training/replay/training_outcome_port_explicit.v2.json"
            )
        );
    }

    #[test]
    fn absent_native_caches_require_bundle_and_graph_proof_of_no_oof_dependency() {
        let bundle = empty_bundle();
        let graph = cache_free_graph();
        let (mut outcome, package) = role_transport_fixture(true);
        outcome.portable_prediction_caches = None;
        let check = |bundle: &ExecutionBundle, graph: &GraphSpec| {
            let mut candidate = outcome.clone();
            candidate.execution_bundle = bundle.clone();
            candidate.effective_plan.graph_plan.graph = graph.clone();
            // Isolate the private eligibility precheck, as the former test did.
            // Full signed-boundary coverage is retained in the test below.
            native_predictor(&candidate, &package)
        };
        check(&bundle, &graph).unwrap();

        let requirement: BundlePredictionRequirement = serde_json::from_value(json!({
            "producer_node": "model:source",
            "source_port": "oof",
            "consumer_node": "model:consumer",
            "target_port": "meta",
            "partition": "validation",
            "prediction_level": "sample",
            "fold_ids": ["fold:0"],
            "sample_ids": ["sample:1"],
            "prediction_width": 1,
            "target_names": ["y"],
        }))
        .expect("test requirement deserializes");
        requirement
            .validate()
            .expect("test requirement is a valid nonempty OOF dependency");
        let mut with_requirement = bundle.clone();
        with_requirement.prediction_requirements.push(requirement);
        assert!(check(&with_requirement, &graph)
            .unwrap_err()
            .to_string()
            .contains("OOF cache dependency"));

        let cache: BundlePredictionCacheRecord = serde_json::from_value(json!({
            "requirement_key": "model:source.oof->model:consumer.meta",
            "cache_id": "prediction-cache:test",
            "format": "dag-ml-json-prediction-blocks-v2",
            "partition": "validation",
            "prediction_level": "sample",
            "fold_ids": ["fold:0"],
            "sample_ids": ["sample:1"],
            "prediction_width": 1,
            "target_names": ["y"],
            "block_count": 1,
            "row_count": 1,
            "content_fingerprint": "d".repeat(64),
            "blocks": [{
                "prediction_id": "prediction:source.fold0",
                "fold_id": "fold:0",
                "prediction_level": "sample",
                "row_count": 1,
                "sample_ids": ["sample:1"],
                "content_fingerprint": "e".repeat(64),
            }],
        }))
        .expect("test cache record deserializes");
        cache
            .validate()
            .expect("test cache record is a valid nonempty OOF dependency");
        let mut with_cache = bundle.clone();
        with_cache.prediction_caches.push(cache);
        assert!(check(&with_cache, &graph)
            .unwrap_err()
            .to_string()
            .contains("OOF cache dependency"));

        let oof_graph: GraphSpec = serde_json::from_str(include_str!(
            "../tests/fixtures/package/separation_branch_concat_merge_oof_graph.json"
        ))
        .expect("OOF graph fixture deserializes");
        assert!(oof_graph
            .edges
            .iter()
            .any(|edge| edge.contract.requires_oof));
        assert!(check(&bundle, &oof_graph)
            .unwrap_err()
            .to_string()
            .contains("OOF cache dependency"));
    }

    #[test]
    fn native_cache_absence_refuses_resigned_oof_with_cleared_bundle_arrays() {
        let (mut stripped, _) = role_transport_fixture(true);
        assert!(stripped
            .effective_plan
            .graph_plan
            .graph
            .edges
            .iter()
            .any(|edge| edge.contract.requires_oof));
        assert!(!stripped.execution_bundle.prediction_requirements.is_empty());
        assert!(!stripped.execution_bundle.prediction_caches.is_empty());

        // This models a malicious but re-signed portable boundary: each
        // ordinary bundle list is cleared, along with refit references and
        // the durable payload set.  The outcome remains structurally valid
        // because normal cross-link validation is deliberately one-way.
        stripped.execution_bundle.prediction_requirements.clear();
        stripped.execution_bundle.prediction_caches.clear();
        for artifact in &mut stripped.execution_bundle.refit_artifacts {
            artifact.prediction_requirement_keys.clear();
        }
        stripped.portable_prediction_caches = None;
        resign_outcome(&mut stripped);
        stripped
            .validate()
            .expect("re-signed fixture reaches the native persistence boundary");

        let package = stripped
            .to_portable_predictor_package(
                "predictor:stripped-oof",
                FittedArtifactMode::PortableRequired,
                ArtifactLoadMode::NativePortable,
            )
            .unwrap();
        let error = predictor_facets(&stripped, &package)
            .expect_err("graph-wide requires_oof forbids absent retained caches");
        assert!(error.to_string().contains("OOF cache dependency"));
    }

    #[test]
    fn native_facets_refuse_stripped_oof_on_write_and_selected_read() {
        let (mut outcome, _) = role_transport_fixture(true);
        assert!(outcome
            .effective_plan
            .graph_plan
            .graph
            .edges
            .iter()
            .any(|edge| edge.contract.requires_oof));
        outcome.execution_bundle.prediction_requirements.clear();
        outcome.execution_bundle.prediction_caches.clear();
        for artifact in &mut outcome.execution_bundle.refit_artifacts {
            artifact.prediction_requirement_keys.clear();
        }
        outcome.portable_prediction_caches = None;
        resign_outcome(&mut outcome);
        outcome.validate().unwrap();
        let package = outcome
            .to_portable_predictor_package(
                "package:stripped-oof",
                FittedArtifactMode::PortableRequired,
                ArtifactLoadMode::NativePortable,
            )
            .unwrap();
        let outcome_bytes = crate::stable_json::to_vec(&outcome).unwrap();
        let package_bytes = crate::stable_json::to_vec(&package).unwrap();
        let error =
            crate::build_native_predictor_facets(&outcome_bytes, &package_bytes).unwrap_err();
        assert!(error.to_string().contains("OOF cache dependency"));

        // A loose facet export and correct raw inventory cannot bypass the
        // native boundary by stripping caches before re-signing both roots.
        let loose =
            crate::build_training_artifact_facets_from_bytes(&outcome_bytes, &package_bytes)
                .unwrap();
        struct Members(BTreeMap<String, Vec<u8>>);
        impl crate::FacetMemberSource for Members {
            fn read_member(&self, path: &str) -> crate::FacetResult<Vec<u8>> {
                Ok(self.0[path].clone())
            }
        }
        let error =
            crate::validate_selected_native_closure(&loose.graph, "model", &Members(loose.members))
                .unwrap_err();
        assert!(error.to_string().contains("OOF cache dependency"));
    }

    #[test]
    fn retained_nonempty_native_cache_keeps_historical_serialization() {
        let mut outcome = TrainingOutcome::from_json(include_str!(
            "../tests/fixtures/package/archive/training_outcome_port_explicit.json"
        ))
        .expect("real V2 stacking outcome fixture validates");
        let retained = outcome
            .portable_prediction_caches
            .as_ref()
            .expect("fixture retains its nonempty OOF cache payload set");
        assert!(!retained.caches.is_empty());

        // Make the existing fixture's refit artifacts portable raw N4MM
        // members without changing its OOF cache contracts.  Native facets
        // bind raw bytes; the test needs no native runtime to prove that a
        // supplied `Some(nonempty)` cache set follows the untouched historical
        // serialization path.
        outcome.execution_bundle.raw_artifact_payloads.clear();
        for (index, record) in outcome
            .execution_bundle
            .refit_artifacts
            .iter_mut()
            .enumerate()
        {
            let payload = format!("n4mm retained-cache fixture {index}").into_bytes();
            let fingerprint = sha256(&payload);
            record.artifact.kind = "n4m_model".to_string();
            record.artifact.backend = Some(ArtifactBackend::Raw);
            record.artifact.uri = Some(format!("methods/retained-cache-{index}.n4mm"));
            record.artifact.content_fingerprint = Some(fingerprint);
            record.artifact.size_bytes = Some(payload.len() as u64);
            record.artifact.plugin = None;
            record.artifact.plugin_version = None;
            record.artifact.abi_major = Some(crate::hpo::METHODS_ABI_MAJOR);
            record.artifact.abi_min_minor = Some(crate::hpo::METHODS_PLS_N4MM_MIN_ABI_MINOR);
            outcome
                .execution_bundle
                .raw_artifact_payloads
                .insert(record.artifact.id.clone(), payload);
        }
        let portable_artifacts = outcome
            .execution_bundle
            .refit_artifacts
            .iter()
            .map(|record| (record.artifact.id.clone(), record.artifact.clone()))
            .collect::<BTreeMap<_, _>>();
        for record in &mut outcome.lineage {
            for artifact in &mut record.artifact_refs {
                if let Some(portable) = portable_artifacts.get(&artifact.id) {
                    *artifact = portable.clone();
                }
            }
        }
        resign_outcome(&mut outcome);
        outcome
            .validate()
            .expect("portable raw fixture remains a valid re-signed outcome");
        let package = outcome
            .to_portable_predictor_package(
                "predictor:archive.retained-nonempty",
                FittedArtifactMode::PortableRequired,
                ArtifactLoadMode::NativePortable,
            )
            .expect("portable raw fixture produces a PREDICT package");
        let export = predictor_facets(&outcome, &package)
            .expect("native facets preserve an existing retained nonempty cache set");
        let closure = selected(&export).unwrap();
        assert!(closure
            .methods_requirements()
            .iter()
            .all(|entry| entry.artifact_kind == "n4m_model"
                && entry.abi_major == crate::hpo::METHODS_ABI_MAJOR
                && entry.abi_min_minor == crate::hpo::METHODS_PLS_N4MM_MIN_ABI_MINOR));
        assert_eq!(
            export.members.get("dagml/prediction.json").unwrap(),
            crate::stable_json::to_vec(outcome.portable_prediction_caches.as_ref().unwrap())
                .unwrap()
                .as_slice(),
            "Some(nonempty) retains the exact historical owner cache JSON bytes"
        );
        assert_eq!(PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION, 2);
    }
}
