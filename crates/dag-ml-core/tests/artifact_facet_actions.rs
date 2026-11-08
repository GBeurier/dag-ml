//! Live owner ports. Metadata-only test adapters below are isolation controls,
//! not qualification of IO numeric projection or scientific dataset identity.
use std::collections::BTreeMap;

use dag_ml_core::*;

const OUTCOME: &[u8] = include_bytes!("fixtures/artifact_facets/native_role/training_outcome.json");
const PACKAGE: &[u8] =
    include_bytes!("fixtures/artifact_facets/native_role/predictor_package.json");
const RELATIONS: &[u8] =
    include_bytes!("fixtures/artifact_facets/native_role/source_relations.json");

struct MetadataOnlyCohort(BTreeMap<String, ExternalDataPlanEnvelope>);
impl FacetReplayCohort for MetadataOnlyCohort {
    fn envelopes(&self) -> &BTreeMap<String, ExternalDataPlanEnvelope> {
        &self.0
    }
}

fn source() -> ArtifactFacetExport {
    build_training_artifact_facets_from_bytes(OUTCOME, PACKAGE).unwrap()
}

fn package() -> PortablePredictorPackage {
    PortablePredictorPackage::from_json(std::str::from_utf8(PACKAGE).unwrap()).unwrap()
}

// Current genuine owner requirements/relations, with a synthetic metadata-only
// inference cohort for default preflight isolation. No claimed numeric IO proof.
fn metadata_cohort(package: &PortablePredictorPackage) -> MetadataOnlyCohort {
    let relations: SampleRelationSet = serde_json::from_slice(RELATIONS).unwrap();
    let inference = SampleRelationSet {
        records: ["inference:new:02", "inference:new:01"]
            .into_iter()
            .enumerate()
            .map(|(index, id)| {
                SampleRelation::new(
                    ObservationId::new(format!("observation:inference:{index}")).unwrap(),
                    SampleId::new(id).unwrap(),
                )
            })
            .collect(),
    };
    let target_names = package.output_bindings[0].target_names.clone();
    MetadataOnlyCohort(
        package
            .execution_bundle
            .data_requirements
            .iter()
            .map(|requirement| {
                let cohort = PredictCohortConstructionRequest {
                    role: PredictCohortRole::Inference,
                    relations: inference.clone(),
                    target_names: target_names.clone(),
                    data_content_fingerprint: "a".repeat(64),
                    target_content_fingerprint: None,
                }
                .derive()
                .unwrap();
                let envelope = ExternalDataPlanEnvelope {
                    schema_version: 2,
                    schema_fingerprint: requirement.schema_fingerprint.clone(),
                    plan_fingerprint: requirement.plan_fingerprint.clone(),
                    relation_fingerprint: Some(relations.fingerprint().unwrap()),
                    data_content_fingerprint: Some("a".repeat(64)),
                    target_content_fingerprint: None,
                    coordinator_relations: Some(relations.clone()),
                    predict_cohort: Some(cohort),
                };
                envelope.validate().unwrap();
                (requirement.key(), envelope)
            })
            .collect(),
    )
}

fn request(
    package: &PortablePredictorPackage,
    cohort: &impl FacetReplayCohort,
) -> TrainingReplayRequest {
    let mut request = TrainingReplayRequest {
        schema_version: 1,
        request_id: "request:facet:predict".into(),
        source_outcome_fingerprint: package.training_outcome.outcome_fingerprint.clone(),
        phase: Phase::Predict,
        data_envelope_keys: cohort.envelopes().keys().cloned().collect(),
        output_binding_ids: package
            .output_bindings
            .iter()
            .map(|b| b.binding_id.clone())
            .collect(),
        request_fingerprint: String::new(),
    };
    request.output_binding_ids.sort();
    request.request_fingerprint = request.compute_fingerprint().unwrap();
    request
}

fn initial_source() -> (
    ArtifactFacetExport,
    InitialFullRefitPackage,
    MetadataOnlyCohort,
) {
    let package = InitialFullRefitPackage::from_json(include_str!(
        "fixtures/artifact_facets/initial_refit.json"
    ))
    .unwrap();
    let export = build_initial_refit_artifact_facets(&package).unwrap();
    let mut envelope = package.training_envelope.clone();
    envelope.schema_version = 2;
    envelope.data_content_fingerprint = Some("a".repeat(64));
    envelope.target_content_fingerprint = None;
    envelope.predict_cohort = metadata_cohort(&self::package())
        .0
        .into_values()
        .next()
        .unwrap()
        .predict_cohort;
    let cohort = MetadataOnlyCohort(
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
    );
    (export, package, cohort)
}

#[test]
fn initial_refit_admission_uses_distinct_owner_output_preflight() {
    let (export, package, cohort) = initial_source();
    let output_ids = package
        .outputs
        .iter()
        .map(|output| output.output_id.clone())
        .collect();
    admit_facet_predict(
        &export.graph,
        &export.context,
        &export.members,
        "full_refit",
        FacetPredictRequest::InitialRefit { output_ids },
        cohort,
    )
    .unwrap();
    let (export, _, cohort) = initial_source();
    assert!(matches!(
        admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "full_refit",
            FacetPredictRequest::InitialRefit {
                output_ids: vec!["output:absent".into()]
            },
            cohort
        ),
        Err(FacetRefusal::OwnerValidation { .. })
    ));
}

#[test]
fn default_admission_moves_target_free_new_cohort_and_actual_root() {
    let export = source();
    let package = package();
    let cohort = metadata_cohort(&package);
    let request = request(&package, &cohort);
    let admitted = admit_facet_predict(
        &export.graph,
        &export.context,
        &export.members,
        "model",
        FacetPredictRequest::Replay(request.clone()),
        cohort,
    )
    .unwrap();
    assert_eq!(
        admitted.root_identity(),
        export.context.owner_proofs["model"].identity()
    );
    let FacetPredictRequest::Replay(actual) = admitted.request() else {
        panic!("replay request");
    };
    assert_eq!(actual, &request);
    for envelope in admitted.envelopes().values() {
        let cohort = envelope.predict_cohort.as_ref().unwrap();
        assert!(cohort.target_content_fingerprint.is_none());
        assert_eq!(
            cohort
                .physical_sample_ids
                .iter()
                .map(SampleId::as_str)
                .collect::<Vec<_>>(),
            ["inference:new:01", "inference:new:02"]
        );
    }
    assert!(
        facet_calibration_view(&export.context.owner_proofs["model"])
            .unwrap()
            .is_none()
    );
}

#[test]
fn admission_refuses_supplied_state_and_non_model_support_before_request() {
    for id in [
        "model",
        "recipe",
        "result",
        "bundle",
        "plan",
        "prediction",
        "score",
    ] {
        let mut export = source();
        let artifact = export.graph.artifacts.iter().find(|a| a.id == id).unwrap();
        let ArtifactFacetStorage::Embedded { root_member, .. } = &artifact.storage else {
            panic!("embedded");
        };
        export.members.get_mut(root_member).unwrap().push(b' ');
        let package = package();
        let cohort = metadata_cohort(&package);
        let request = request(&package, &cohort);
        assert!(matches!(
            admit_facet_predict(
                &export.graph,
                &export.context,
                &export.members,
                "model",
                FacetPredictRequest::Replay(request),
                cohort
            ),
            Err(FacetRefusal::Mismatch { .. })
        ));
    }
    let mut export = source();
    let model = export
        .graph
        .artifacts
        .iter()
        .find(|a| a.id == "model")
        .unwrap();
    let states = validate_facet_members(
        model,
        &export.context.owner_proofs["model"],
        &export.members,
    )
    .unwrap();
    export.members.get_mut(&states[0].member_path).unwrap()[0] ^= 1;
    let package = package();
    let cohort = metadata_cohort(&package);
    let request = request(&package, &cohort);
    assert!(matches!(
        admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(request),
            cohort
        ),
        Err(FacetRefusal::Mismatch { .. })
    ));
}

#[test]
fn admission_refuses_changed_source_binding_keys_and_explain_without_runtime() {
    for mutation in 0..5 {
        let export = source();
        let package = package();
        let mut cohort = metadata_cohort(&package);
        let mut request = request(&package, &cohort);
        match mutation {
            0 => request.source_outcome_fingerprint = "f".repeat(64),
            1 => request.output_binding_ids = vec!["binding:absent".into()],
            2 => {
                let key = cohort.0.keys().next().unwrap().clone();
                cohort.0.remove(&key);
            }
            3 => request.phase = Phase::Explain,
            4 => {
                for envelope in cohort.0.values_mut() {
                    let current = envelope.predict_cohort.as_mut().unwrap();
                    current.target_names = vec!["wrong-target".into()];
                    current.cohort_fingerprint = current.fingerprint().unwrap();
                }
            }
            _ => unreachable!(),
        }
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        let result = admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(request),
            cohort,
        );
        match mutation {
            0 | 1 => assert!(matches!(result, Err(FacetRefusal::OwnerValidation { .. }))),
            2 => assert!(matches!(
                result,
                Err(FacetRefusal::InvalidStructure { .. } | FacetRefusal::Mismatch { .. })
            )),
            3 => assert!(matches!(
                result,
                Err(FacetRefusal::UnsupportedOperation { .. })
            )),
            4 => assert!(matches!(result, Err(FacetRefusal::Mismatch { .. }))),
            _ => unreachable!(),
        }
    }
}

#[cfg(feature = "methods-optimizer")]
mod native {
    use super::*;
    use sha2::Digest;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    // A deliberately synthetic native isolation adapter, NOT real IO/DMD
    // qualification. The external owner harness must cover that separate port.
    struct SyntheticSnapshot {
        envelopes: BTreeMap<String, ExternalDataPlanEnvelope>,
        data: MethodsPlsDataset,
        projections: Arc<AtomicUsize>,
    }
    impl MethodsFacetCohortOwner for SyntheticSnapshot {
        fn into_bindings(self) -> FacetResult<BTreeMap<String, MethodsFacetBinding>> {
            self.projections.fetch_add(1, Ordering::SeqCst);
            Ok(self
                .envelopes
                .into_iter()
                .map(|(key, envelope)| {
                    (
                        key,
                        MethodsFacetBinding {
                            envelope,
                            dataset: self.data.clone(),
                        },
                    )
                })
                .collect())
        }
    }

    fn runtime() -> MethodsRuntime {
        let library =
            std::env::var("N4M_LIBRARY_PATH").expect("native test requires actual owner runtime");
        assert_eq!(std::env::var("N4M_LIB_PATH").unwrap(), library);
        let expected = std::env::var("N4M_EXPECTED_LIBRARY_SHA256")
            .expect("native test requires the qualified owner library SHA256");
        assert_eq!(expected.len(), 64);
        assert!(expected
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)));
        assert_eq!(
            format!(
                "{:x}",
                sha2::Sha256::digest(std::fs::read(&library).unwrap())
            ),
            expected
        );
        MethodsRuntime::configure(&library).unwrap()
    }

    fn data() -> MethodsPlsDataset {
        MethodsPlsDataset {
            sample_ids: ["inference:new:02", "inference:new:01"]
                .into_iter()
                .map(|id| SampleId::new(id).unwrap())
                .collect(),
            x: MethodsPlsMatrix {
                rows: 2,
                cols: 2,
                values: vec![3.0, 9.0, 1.0, 1.0],
            },
            y: None,
            target_names: vec!["protein".into()],
            y_validity_masks: None,
        }
    }

    fn admitted(
        data: MethodsPlsDataset,
        projections: Arc<AtomicUsize>,
    ) -> AdmittedFacetReplay<FrozenMethodsFacetCohort> {
        let package = package();
        let export = source();
        let cohort = freeze_methods_facet_cohort(SyntheticSnapshot {
            envelopes: metadata_cohort(&package).0,
            data,
            projections,
        })
        .unwrap();
        let request = request(&package, &cohort);
        admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(request),
            cohort,
        )
        .unwrap()
    }

    fn predict(
        admitted: AdmittedFacetReplay<FrozenMethodsFacetCohort>,
        label: &str,
    ) -> TrainingReplayOutcome {
        let outcome = execute_admitted_methods_facet_replay(
            admitted,
            runtime(),
            format!("outcome:facet:{label}"),
            RunId::new(format!("run:facet:{label}")).unwrap(),
        )
        .unwrap();
        let FacetMethodsReplayOutcome::Predictor(outcome) = outcome else {
            panic!("predictor replay");
        };
        *outcome
    }

    // Real numeric rows and freshly trained owner artifacts, not edited saved
    // outcomes. This is a DAG/controller test provider, not an IO identity proof.
    struct TargetRows {
        inner: EnvelopeAttestedRuntimeDataProvider<dag_ml_core::data::InMemoryDataProvider>,
        data: MethodsPlsDataset,
    }

    impl RuntimeDataProvider for TargetRows {
        fn materialize(&self, request: &DataMaterializationRequest) -> Result<HandleRef> {
            self.inner.materialize(request)
        }

        fn make_view(&self, request: &DataViewRequest) -> Result<HandleRef> {
            self.inner.make_view(request)
        }

        fn training_data_identity(
            &self,
            binding: &DataBinding,
        ) -> Result<Option<TrainingDataIdentity>> {
            self.inner.training_data_identity(binding)
        }

        fn coordinator_relations(
            &self,
            binding: &DataBinding,
        ) -> Result<Option<SampleRelationSet>> {
            self.inner.coordinator_relations(binding)
        }

        fn methods_pls_capability(&self) -> Result<()> {
            Ok(())
        }

        fn methods_pls_data(&self, request: &MethodsPlsDataRequest) -> Result<MethodsPlsData> {
            request.validate()?;
            let rows = |view: &DataProviderViewSpec| -> Result<MethodsPlsDataset> {
                let ids = view
                    .sample_ids
                    .as_ref()
                    .expect("training views select actual IDs");
                let indices = ids
                    .iter()
                    .map(|id| {
                        self.data
                            .sample_ids
                            .iter()
                            .position(|source| source == id)
                            .ok_or_else(|| {
                                DagMlError::RuntimeValidation(format!("unknown test row {id}"))
                            })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let select = |matrix: &MethodsPlsMatrix| MethodsPlsMatrix {
                    rows: ids.len(),
                    cols: matrix.cols,
                    values: indices
                        .iter()
                        .flat_map(|index| {
                            matrix.values[index * matrix.cols..(index + 1) * matrix.cols]
                                .iter()
                                .copied()
                        })
                        .collect(),
                };
                Ok(MethodsPlsDataset {
                    sample_ids: ids.clone(),
                    x: select(&self.data.x),
                    y: self.data.y.as_ref().map(select),
                    target_names: self.data.target_names.clone(),
                    y_validity_masks: None,
                })
            };
            Ok(MethodsPlsData {
                fit: rows(&request.fit_view)?,
                prediction: request.prediction_view.as_ref().map(rows).transpose()?,
            })
        }
    }

    fn train_target_model(
        target_index: Option<usize>,
        classifier: bool,
    ) -> (TrainingOutcome, PortablePredictorPackage, MethodsPlsDataset) {
        let runtime = runtime();
        let registry = derive_host_controller_registry(
            &methods_estimator_host_controller_specs(&runtime).unwrap(),
        )
        .unwrap();
        let owner = if classifier {
            "controller:n4m.classifier"
        } else {
            "controller:n4m.regressor"
        };
        let manifest = registry
            .manifests()
            .find(|manifest| manifest.controller_id.as_str() == owner)
            .expect("actual Methods manifest supplies this controller")
            .clone();
        let mut request = TrainingRequest::from_json(include_str!(
            "fixtures/artifact_facets/native_role/source_training_request.json"
        ))
        .unwrap();
        let source_rows: BTreeMap<SampleId, [f64; 4]> = serde_json::from_str(include_str!(
            "fixtures/artifact_facets/native_role/source_provider_rows.json"
        ))
        .unwrap();
        let names: Vec<String> = if classifier {
            vec!["class".into()]
        } else {
            vec!["protein".into(), "moisture".into()]
        };
        let second_target = [3.0, 4.0, 8.0, 12.0, 18.0, 25.0, 33.0, 42.0];
        let data = MethodsPlsDataset {
            sample_ids: source_rows.keys().cloned().collect(),
            x: MethodsPlsMatrix {
                rows: 8,
                cols: 2,
                values: source_rows
                    .values()
                    .flat_map(|row| [row[0], row[1]])
                    .collect(),
            },
            y: Some(MethodsPlsMatrix {
                rows: 8,
                cols: names.len(),
                values: source_rows
                    .values()
                    .enumerate()
                    .flat_map(|(index, row)| {
                        if classifier {
                            vec![if index % 2 == 0 { 10.0 } else { 20.0 }]
                        } else {
                            vec![row[3], second_target[index]]
                        }
                    })
                    .collect(),
            }),
            target_names: names.clone(),
            y_validity_masks: None,
        };
        data.validate("actual owner training rows", true).unwrap();
        let model = &mut request.graph.nodes[0];
        let method = if classifier {
            "models.classification.pls_logistic"
        } else {
            "models.pls.pls_regression"
        };
        model.operator = Some(serde_json::json!(format!("n4m:{method}")));
        model.params = BTreeMap::from([
            ("method_id".into(), serde_json::json!(method)),
            ("n_components".into(), serde_json::json!(1)),
        ]);
        if let Some(index) = target_index {
            model
                .params
                .insert("native_target_index".into(), serde_json::json!(index));
        }
        model
            .metadata
            .insert("controller_id".into(), serde_json::json!(owner));
        model.ports.outputs = manifest.output_ports.clone();
        request.controller_manifests = vec![manifest];
        let output = &mut request.options.outputs[0];
        output.port_name = Some("y_hat".into());
        output.target_names =
            target_index.map_or_else(|| names.clone(), |index| vec![names[index].clone()]);
        output.target_units = vec![None; output.target_names.len()];
        output.class_labels = if classifier {
            vec![vec!["10".into(), "20".into()]]
        } else {
            vec![Vec::new(); output.target_names.len()]
        };
        output.prediction_kind = if classifier {
            PredictionKind::ClassLabel
        } else {
            PredictionKind::RegressionPoint
        };
        if classifier {
            request.options.selection.metric = serde_json::from_value(
                serde_json::json!({"name":"accuracy","objective":"maximize"}),
            )
            .unwrap();
        }
        let relations: SampleRelationSet = serde_json::from_slice(RELATIONS).unwrap();
        let x_fingerprint = methods_pls_predict_feature_content_fingerprint(&data.x).unwrap();
        // Test-provider target content is the actual serialized named target
        // matrix. TrainingDataIdentity is then built by the public owner API.
        let y_fingerprint = format!(
            "{:x}",
            sha2::Sha256::digest(
                serde_json::to_vec(&(&names, &data.y.as_ref().unwrap().values)).unwrap()
            )
        );
        let bindings = request
            .campaign
            .data_bindings
            .values()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let envelopes = bindings
            .iter()
            .map(|binding| {
                (
                    data_binding_requirement_key(&binding.node_id, &binding.input_name),
                    ExternalDataPlanEnvelope {
                        schema_version: 1,
                        schema_fingerprint: binding.schema_fingerprint.clone(),
                        plan_fingerprint: binding.plan_fingerprint.clone(),
                        relation_fingerprint: Some(relations.fingerprint().unwrap()),
                        data_content_fingerprint: Some(x_fingerprint.clone()),
                        target_content_fingerprint: Some(y_fingerprint.clone()),
                        coordinator_relations: Some(relations.clone()),
                        predict_cohort: None,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        request.data_identities = bindings
            .iter()
            .map(|binding| {
                TrainingDataIdentity::from_binding_envelope(
                    binding,
                    &envelopes
                        [&data_binding_requirement_key(&binding.node_id, &binding.input_name)],
                )
                .unwrap()
            })
            .collect();
        request
            .data_identities
            .sort_by(|left, right| left.requirement_key.cmp(&right.requirement_key));
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        request.validate().unwrap();
        let mut raw = dag_ml_core::data::InMemoryDataProvider::new(
            ControllerId::new("controller:facet.actual-training-rows").unwrap(),
        );
        for envelope in envelopes.values() {
            raw.register_envelope(envelope.clone()).unwrap();
        }
        let provider = TargetRows {
            inner: EnvelopeAttestedRuntimeDataProvider::new(raw, bindings, envelopes).unwrap(),
            data: data.clone(),
        };
        let influence: TrainingInfluenceManifest = serde_json::from_str(include_str!(
            "fixtures/artifact_facets/native_role/source_training_influence.json"
        ))
        .unwrap();
        let mut controllers = RuntimeControllerRegistry::new();
        register_methods_native_controllers(&mut controllers, runtime).unwrap();
        let outcome = execute_training(TrainingExecutionInput {
            request: &request,
            outcome_id: "outcome:facet.actual-target".into(),
            run_id: RunId::new("run:facet.actual-target").unwrap(),
            bundle_id: BundleId::new("bundle:facet.actual-target").unwrap(),
            controllers: &controllers,
            data_provider: &provider,
            relations: &relations,
            training_influence: &influence,
            artifact_store: &mut InMemoryArtifactStore::new(),
            warnings: Vec::new(),
            diagnostics: BTreeMap::new(),
        })
        .expect("actual native CV/SELECT/REFIT must produce the artifacts");
        let package = outcome
            .to_portable_predictor_package(
                "package:facet.actual-target",
                FittedArtifactMode::PortableRequired,
                ArtifactLoadMode::NativePortable,
            )
            .unwrap();
        assert!(package
            .execution_bundle
            .refit_artifacts
            .iter()
            .any(|record| { record.artifact.kind == "n4m_estimator" }));
        (outcome, package, data)
    }

    fn inference_envelopes(
        package: &PortablePredictorPackage,
        rows: &MethodsPlsDataset,
    ) -> BTreeMap<String, ExternalDataPlanEnvelope> {
        assert!(rows.y.is_none());
        let relations = SampleRelationSet {
            records: rows
                .sample_ids
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    SampleRelation::new(
                        ObservationId::new(format!("obs:facet.actual-predict:{index}")).unwrap(),
                        id.clone(),
                    )
                })
                .collect(),
        };
        let content = methods_pls_predict_feature_content_fingerprint(&rows.x).unwrap();
        let cohort = PredictCohortConstructionRequest {
            role: PredictCohortRole::Inference,
            relations: relations.clone(),
            target_names: package.output_bindings[0].target_names.clone(),
            data_content_fingerprint: content.clone(),
            target_content_fingerprint: None,
        }
        .derive()
        .unwrap();
        package
            .execution_bundle
            .data_requirements
            .iter()
            .map(|requirement| {
                (
                    requirement.key(),
                    ExternalDataPlanEnvelope {
                        schema_version: 2,
                        schema_fingerprint: requirement.schema_fingerprint.clone(),
                        plan_fingerprint: requirement.plan_fingerprint.clone(),
                        relation_fingerprint: Some(relations.fingerprint().unwrap()),
                        data_content_fingerprint: Some(content.clone()),
                        target_content_fingerprint: None,
                        coordinator_relations: Some(relations.clone()),
                        predict_cohort: Some(cohort.clone()),
                    },
                )
            })
            .collect()
    }

    #[test]
    fn selected_target_indices_zero_and_one_match_actual_wide_owner_replay_bits() {
        for index in [0, 1] {
            println!("selected-target case index={index}: genuine training and wide reference");
            let (outcome, package, training) = train_target_model(Some(index), false);
            let export = build_training_artifact_facets(&outcome, &package).unwrap();
            let mut wide = data();
            wide.target_names = training.target_names;
            let envelopes = inference_envelopes(&package, &wide);
            let mut narrow = wide.clone();
            narrow.target_names = package.output_bindings[0].target_names.clone();
            assert_eq!(narrow.target_names, vec![wide.target_names[index].clone()]);
            let projections = Arc::new(AtomicUsize::new(0));
            let frozen = freeze_methods_facet_cohort(SyntheticSnapshot {
                envelopes: envelopes.clone(),
                data: narrow,
                projections: projections.clone(),
            })
            .unwrap();
            let request = request(&package, &frozen);
            let admitted = admit_facet_predict(
                &export.graph,
                &export.context,
                &export.members,
                "model",
                FacetPredictRequest::Replay(request.clone()),
                frozen,
            )
            .unwrap();
            // The current legitimate wide route projects target_index once.
            // It is an independent direct owner reference, not a weakened
            // replacement for the narrowed frozen-cohort action under test.
            let inputs = envelopes
                .keys()
                .map(|key| (key.clone(), wide.clone()))
                .collect();
            let reference =
                execute_loaded_methods_predictor_replay(MethodsPortablePredictorReplayInput {
                    package: &package,
                    request: &request,
                    data_envelopes: &envelopes,
                    methods_inputs: &inputs,
                    runtime: runtime(),
                    outcome_id: format!("outcome:facet.wide:{index}"),
                    run_id: RunId::new(format!("run:facet.wide:{index}")).unwrap(),
                    warnings: Vec::new(),
                    diagnostics: BTreeMap::new(),
                })
                .expect("the existing wide target route remains valid");
            let actual = predict(admitted, &format!("selected-{index}"));
            assert_eq!(projections.load(Ordering::SeqCst), 1);
            let actual = &actual.outputs[0].predictions[0];
            let reference = &reference.outputs[0].predictions[0];
            assert_eq!(actual.sample_ids, reference.sample_ids);
            assert_eq!(
                actual
                    .sample_ids
                    .iter()
                    .map(SampleId::as_str)
                    .collect::<Vec<_>>(),
                ["inference:new:01", "inference:new:02"]
            );
            assert_eq!(actual.target_names, reference.target_names);
            assert_eq!(actual.target_names, package.output_bindings[0].target_names);
            assert_eq!(
                actual
                    .values
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                reference
                    .values
                    .iter()
                    .flatten()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            println!("selected-target case index={index}: admitted replay bits match wide owner");
        }
    }

    #[test]
    fn genuine_multi_target_regression_still_passes_calibration_preflight() {
        let (outcome, package, _) = train_target_model(None, false);
        assert_eq!(package.output_bindings[0].target_names.len(), 2);
        assert_eq!(
            package.output_bindings[0].prediction_kind,
            PredictionKind::RegressionPoint
        );
        let export = build_training_artifact_facets(&outcome, &package).unwrap();
        preflight_facet_calibration(
            &export.graph,
            &export.context,
            &export.members,
            "result",
            "model",
            &package.output_bindings[0].binding_id,
        )
        .unwrap();
    }

    #[test]
    fn genuine_classifier_calibration_refuses_before_runtime_in_fresh_process() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let (outcome, package, _) = train_target_model(None, true);
        assert_eq!(
            package.output_bindings[0].prediction_kind,
            PredictionKind::ClassLabel
        );
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "native::classifier_calibration_without_runtime_child",
                "--ignored",
                "--nocapture",
            ])
            .env_remove("N4M_LIB_PATH")
            .env_remove("N4M_LIBRARY_PATH")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        println!("classifier static preflight child PID {}", child.id());
        let bytes =
            serde_json::to_vec(&serde_json::json!({"outcome":outcome,"package":package})).unwrap();
        child.stdin.take().unwrap().write_all(&bytes).unwrap();
        let result = child.wait_with_output().unwrap();
        print!("{}", String::from_utf8_lossy(&result.stdout));
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(String::from_utf8_lossy(&result.stdout)
            .contains("classifier preflight: UnsupportedOperation"));
        #[cfg(target_os = "linux")]
        assert!(String::from_utf8_lossy(&result.stdout).contains("no libn4m mapping"));
    }

    #[test]
    #[ignore = "fresh subprocess control supplied genuine owner-produced artifacts on stdin"]
    fn classifier_calibration_without_runtime_child() {
        use std::io::Read;
        assert!(std::env::var_os("N4M_LIB_PATH").is_none());
        assert!(std::env::var_os("N4M_LIBRARY_PATH").is_none());
        let no_runtime = || {
            #[cfg(target_os = "linux")]
            assert!(!std::fs::read_to_string("/proc/self/maps")
                .unwrap()
                .contains("libn4m"));
        };
        no_runtime();
        let mut bytes = Vec::new();
        std::io::stdin().read_to_end(&mut bytes).unwrap();
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct TrainedArtifacts {
            outcome: TrainingOutcome,
            package: PortablePredictorPackage,
        }
        let source: TrainedArtifacts = serde_json::from_slice(&bytes).unwrap();
        assert!(source.outcome.conformal_calibration.is_none());
        let export = build_training_artifact_facets(&source.outcome, &source.package).unwrap();
        export.validate().unwrap();
        let binding = &source.package.output_bindings[0];
        assert_eq!(binding.prediction_kind, PredictionKind::ClassLabel);
        let refusal = preflight_facet_calibration(
            &export.graph,
            &export.context,
            &export.members,
            "result",
            "model",
            &binding.binding_id,
        )
        .unwrap_err();
        match refusal {
            FacetRefusal::UnsupportedOperation { operation } => {
                assert_eq!(operation, "calibrate a non-regression-point output binding");
            }
            other => panic!("wrong classifier preflight refusal: {other:?}"),
        }
        // The selected binding exists: an earlier unknown-binding refusal is
        // not allowed to make the classifier regression look successful.
        assert!(
            matches!(preflight_facet_calibration(&export.graph, &export.context, &export.members,
            "result", "model", "binding:absent"), Err(FacetRefusal::Mismatch { detail })
            if detail == "requested calibration binding is not a validated package output binding")
        );
        no_runtime();
        println!("classifier preflight: UnsupportedOperation");
        #[cfg(target_os = "linux")]
        println!("no libn4m mapping before or after static preflight");
        #[cfg(not(target_os = "linux"))]
        println!("process mapping observation unavailable on this platform");
    }

    #[test]
    fn actual_native_execution_consumes_frozen_rows_and_reindexes_by_stable_ids() {
        let projections = Arc::new(AtomicUsize::new(0));
        let mut caller = data();
        let frozen = admitted(caller.clone(), projections.clone());
        assert_eq!(projections.load(Ordering::SeqCst), 1);
        // Deep-owned snapshot: later caller data changes have no effect. There
        // is no later buffer/provider argument on the admitted executor.
        caller.x.values.fill(99999.0);
        caller.sample_ids.reverse();
        let first = predict(frozen, "frozen");
        let mut reordered = data();
        reordered.sample_ids.reverse();
        reordered.x.values = vec![1.0, 1.0, 3.0, 9.0];
        let second = predict(admitted(reordered, projections.clone()), "reordered");
        assert_eq!(projections.load(Ordering::SeqCst), 2);
        assert_eq!(first.outputs[0].predictions, second.outputs[0].predictions);
        assert_eq!(
            first.outputs[0].predictions[0]
                .sample_ids
                .iter()
                .map(SampleId::as_str)
                .collect::<Vec<_>>(),
            ["inference:new:01", "inference:new:02"]
        );
        assert!(first
            .input_data_identities
            .iter()
            .all(|identity| identity.target_content_fingerprint.is_none()));
        assert!(first.conformal_intervals.is_empty());
    }

    #[test]
    fn frozen_adapter_refuses_wrong_rows_targets_and_unbound_labels() {
        for mutation in 0..3 {
            let mut data = data();
            match mutation {
                0 => data.sample_ids[0] = SampleId::new("inference:unbound").unwrap(),
                1 => data.target_names = vec!["wrong-target".into()],
                2 => {
                    data.y = Some(MethodsPlsMatrix {
                        rows: 2,
                        cols: 1,
                        values: vec![1.0, 2.0],
                    })
                }
                _ => unreachable!(),
            }
            let result = freeze_methods_facet_cohort(SyntheticSnapshot {
                envelopes: metadata_cohort(&package()).0,
                data,
                projections: Arc::new(AtomicUsize::new(0)),
            });
            assert!(matches!(result, Err(FacetRefusal::Mismatch { .. })));
        }
    }

    #[test]
    fn native_executor_refuses_host_mode_before_any_hydration() {
        let source = TrainingOutcome::from_json(std::str::from_utf8(OUTCOME).unwrap()).unwrap();
        let package = source
            .to_portable_predictor_package(
                "package:facet:host-mode",
                FittedArtifactMode::AllowHostSidecar,
                ArtifactLoadMode::HostSidecar,
            )
            .unwrap();
        let export = build_training_artifact_facets(&source, &package).unwrap();
        let cohort = freeze_methods_facet_cohort(SyntheticSnapshot {
            envelopes: metadata_cohort(&package).0,
            data: data(),
            projections: Arc::new(AtomicUsize::new(0)),
        })
        .unwrap();
        let request = request(&package, &cohort);
        let admitted = admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(request),
            cohort,
        )
        .unwrap();
        assert!(matches!(
            execute_admitted_methods_facet_replay(
                admitted,
                runtime(),
                "outcome:host:refused".into(),
                RunId::new("run:host:refused").unwrap()
            ),
            Err(FacetRefusal::MissingTrust)
        ));
    }

    #[test]
    fn native_executor_refuses_unregistered_initial_controller_before_dispatch() {
        let (export, package, metadata) = initial_source();
        let cohort = freeze_methods_facet_cohort(SyntheticSnapshot {
            envelopes: metadata.0,
            data: data(),
            projections: Arc::new(AtomicUsize::new(0)),
        })
        .unwrap();
        let output_ids = package
            .outputs
            .iter()
            .map(|output| output.output_id.clone())
            .collect();
        let admitted = admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "full_refit",
            FacetPredictRequest::InitialRefit { output_ids },
            cohort,
        )
        .unwrap();
        assert!(matches!(
            execute_admitted_methods_facet_replay(
                admitted,
                runtime(),
                "outcome:initial:refused".into(),
                RunId::new("run:initial:refused").unwrap()
            ),
            Err(FacetRefusal::UnsupportedPayload { .. })
        ));
    }

    #[test]
    fn actual_child_refit_predict_uses_its_own_state_and_target_run() {
        let parent = TrainingOutcome::from_json(std::str::from_utf8(OUTCOME).unwrap()).unwrap();
        let package = package();
        let child = PortableRefitPackageV3::from_json(include_str!(
            "fixtures/artifact_facets/native_role/child_refit_v3.json"
        ))
        .unwrap();
        let relations: SampleRelationSet = serde_json::from_str(include_str!(
            "fixtures/artifact_facets/native_role/target_relations.json"
        ))
        .unwrap();
        let export = build_refit_artifact_facets(&child, &parent, &package).unwrap();
        let bundle = child.outcome.to_runtime_replay_bundle().unwrap();
        let mut envelopes = metadata_cohort(&package).0;
        for requirement in &bundle.data_requirements {
            let envelope = envelopes.get_mut(&requirement.key()).unwrap();
            envelope.schema_fingerprint = requirement.schema_fingerprint.clone();
            envelope.plan_fingerprint = requirement.plan_fingerprint.clone();
            envelope.relation_fingerprint = Some(relations.fingerprint().unwrap());
            envelope.coordinator_relations = Some(relations.clone());
        }
        let replay_envelopes = envelopes.clone();
        let cohort = freeze_methods_facet_cohort(SyntheticSnapshot {
            envelopes,
            data: data(),
            projections: Arc::new(AtomicUsize::new(0)),
        })
        .unwrap();
        let mut request = request(&package, &cohort);
        request.source_outcome_fingerprint = child.outcome.outcome_fingerprint.clone();
        request.output_binding_ids = child
            .outcome
            .output_bindings
            .iter()
            .map(|binding| binding.binding_id.clone())
            .collect();
        request.output_binding_ids.sort();
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        let admitted = admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "full_refit",
            FacetPredictRequest::Replay(request.clone()),
            cohort,
        )
        .unwrap();
        let actual = execute_admitted_methods_facet_replay(
            admitted,
            runtime(),
            "outcome:facet:child".into(),
            RunId::new("run:facet:child").unwrap(),
        )
        .unwrap();
        let FacetMethodsReplayOutcome::ChildRefit(actual) = actual else {
            panic!("child refit replay");
        };
        assert_eq!(
            actual.source_refit_outcome_fingerprint,
            child.outcome.outcome_fingerprint
        );
        assert_eq!(actual.source_package_fingerprint, child.package_fingerprint);
        assert!(actual
            .outputs
            .iter()
            .flat_map(|output| &output.predictions)
            .all(|point| point.partition == PredictionPartition::Final
                && point.sample_ids.len() == 2));

        actual.validate_against(&child, &request).unwrap();
        actual
            .validate_against_current_cohort(&child, &request, &replay_envelopes)
            .unwrap();
        // Synthetic stored metadata mutant of the ACTUAL public-produced V3 replay.
        // Keep a nonempty, intrinsically valid request: an empty one would not
        // exercise bundle coverage on this one-key producer.
        let mut stored = actual;
        let mut edited_request = request;
        let known_key = edited_request.data_envelope_keys[0].clone();
        edited_request.data_envelope_keys[0] = "unknown.input".into();
        edited_request.data_envelope_keys.sort();
        let identity = stored
            .input_data_identities
            .iter_mut()
            .find(|identity| identity.requirement_key == known_key)
            .unwrap();
        identity.requirement_key = "unknown.input".into();
        identity.identity_fingerprint = identity.compute_fingerprint().unwrap();
        stored
            .input_data_identities
            .sort_by(|a, b| a.requirement_key.cmp(&b.requirement_key));
        edited_request.request_fingerprint = edited_request.compute_fingerprint().unwrap();
        edited_request.validate().unwrap();
        assert!(!edited_request.data_envelope_keys.is_empty());
        stored.replay_request_fingerprint = edited_request.request_fingerprint.clone();
        stored.outcome_fingerprint = stored.compute_fingerprint().unwrap();
        let error = stored
            .validate_against(&child, &edited_request)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unknown data envelope key `unknown.input`"),
            "{error}"
        );
        assert!(PortableRefitReplayOutcomeV3::from_json_for_package(
            &serde_json::to_string(&stored).unwrap(),
            &child,
            &edited_request
        )
        .unwrap_err()
        .to_string()
        .contains("unknown data envelope key `unknown.input`"));
        assert!(matches!(
            stored.validate_against_current_cohort(&child, &edited_request, &replay_envelopes),
            Err(ReplayCurrentCohortError::Invalid(_))
        ));
    }

    #[test]
    fn actual_calibrated_native_predict_uses_retained_schema2_state_on_unlabeled_new_rows() {
        let export = build_training_artifact_facets_from_bytes(
            include_bytes!("fixtures/artifact_facets_b1/outputs/training_outcome.after.json"),
            include_bytes!("fixtures/artifact_facets_b1/outputs/ppc.after.json"),
        )
        .unwrap();
        let package = PortablePredictorPackage::from_json(include_str!(
            "fixtures/artifact_facets_b1/outputs/ppc.after.json"
        ))
        .unwrap();
        let mut envelopes: BTreeMap<String, ExternalDataPlanEnvelope> = serde_json::from_str(
            include_str!("fixtures/artifact_facets_b1/outputs/replay_envelope.json"),
        )
        .unwrap();
        let mut data = data();
        data.x.cols = 8;
        data.x.values = vec![
            3.0, 9.0, 2.0, 1.0, 5.0, 2.0, 4.0, 8.0, 1.0, 1.0, 4.0, 3.0, 2.0, 6.0, 7.0, 2.0,
        ];
        let content = methods_pls_predict_feature_content_fingerprint(&data.x).unwrap();
        let relations = SampleRelationSet {
            records: data
                .sample_ids
                .iter()
                .enumerate()
                .map(|(index, id)| {
                    SampleRelation::new(
                        ObservationId::new(format!("obs:facet:calibrated:{index}")).unwrap(),
                        id.clone(),
                    )
                })
                .collect(),
        };
        for envelope in envelopes.values_mut() {
            envelope.predict_cohort = Some(
                PredictCohortConstructionRequest {
                    role: PredictCohortRole::Inference,
                    relations: relations.clone(),
                    target_names: data.target_names.clone(),
                    data_content_fingerprint: content.clone(),
                    target_content_fingerprint: None,
                }
                .derive()
                .unwrap(),
            );
            envelope.data_content_fingerprint = Some(content.clone());
            envelope.target_content_fingerprint = None;
        }
        let cohort = freeze_methods_facet_cohort(SyntheticSnapshot {
            envelopes,
            data,
            projections: Arc::new(AtomicUsize::new(0)),
        })
        .unwrap();
        let request = request(&package, &cohort);
        let admitted = admit_facet_predict(
            &export.graph,
            &export.context,
            &export.members,
            "model",
            FacetPredictRequest::Replay(request),
            cohort,
        )
        .unwrap();
        let actual = predict(admitted, "calibrated");
        assert_eq!(actual.conformal_intervals.len(), 1);
        let interval = &actual.conformal_intervals[0];
        assert_eq!(
            interval.calibration_fingerprint,
            package
                .conformal_calibration
                .as_ref()
                .unwrap()
                .calibration_fingerprint
        );
        assert_eq!(
            interval.sample_ids,
            actual.outputs[0].predictions[0].sample_ids
        );
        assert!(actual
            .input_data_identities
            .iter()
            .all(|identity| identity.target_content_fingerprint.is_none()));
        assert_eq!(actual.source_training_outcome, package.training_outcome);
    }
}

mod calibration {
    use super::*;
    const BEFORE: &[u8] =
        include_bytes!("fixtures/artifact_facets_b1/outputs/training_outcome.before.json");
    const BEFORE_PACKAGE: &[u8] =
        include_bytes!("fixtures/artifact_facets_b1/outputs/ppc.before.json");
    const AFTER: &[u8] =
        include_bytes!("fixtures/artifact_facets_b1/outputs/training_outcome.after.json");
    const AFTER_PACKAGE: &[u8] =
        include_bytes!("fixtures/artifact_facets_b1/outputs/ppc.after.json");
    const REPLAY: &str =
        include_str!("fixtures/artifact_facets_b1/outputs/calibration_replay.json");
    const TRUTH: &str = include_str!("fixtures/artifact_facets_b1/outputs/calibration_truth.json");
    const CAL_RELATIONS: &str =
        include_str!("fixtures/artifact_facets_b1/outputs/calibration_relations.json");

    fn before() -> ArtifactFacetExport {
        build_training_artifact_facets_from_bytes(BEFORE, BEFORE_PACKAGE).unwrap()
    }
    fn after() -> ArtifactFacetExport {
        build_training_artifact_facets_from_bytes(AFTER, AFTER_PACKAGE).unwrap()
    }
    fn calibration_request<'a>(
        replay: &'a TrainingReplayOutcome,
        relations: &'a SampleRelationSet,
        binding: &'a str,
    ) -> FacetCalibrationRequest<'a> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ProducedTruth {
            target_names: Vec<String>,
            sample_ids: Vec<SampleId>,
            values: Vec<Vec<f64>>,
        }
        let truth: ProducedTruth = serde_json::from_str(TRUTH).unwrap();
        assert_eq!(truth.target_names, replay.outputs[0].binding.target_names);
        FacetCalibrationRequest {
            replay,
            binding_id: binding,
            relations,
            truth: ConformalCalibrationTruth {
                sample_ids: truth.sample_ids,
                values: truth.values,
            },
            coverages: vec![0.8, 0.9],
            multi_target_policy: ConformalMultiTargetPolicy::Marginal,
            small_sample_policy: ConformalSmallSamplePolicy::Unbounded,
        }
    }
    fn state_bytes(export: &ArtifactFacetExport) -> BTreeMap<String, Vec<u8>> {
        let model = export
            .graph
            .artifacts
            .iter()
            .find(|a| a.id == "model")
            .unwrap();
        validate_facet_members(
            model,
            &export.context.owner_proofs["model"],
            &export.members,
        )
        .unwrap()
        .into_iter()
        .map(|state| {
            (
                state.member_path.clone(),
                export.members[&state.member_path].clone(),
            )
        })
        .collect()
    }

    #[test]
    fn genuine_regression_calibration_preflight_remains_valid_without_runtime() {
        let export = before();
        let package =
            PortablePredictorPackage::from_json(std::str::from_utf8(BEFORE_PACKAGE).unwrap())
                .unwrap();
        let binding = &package.output_bindings[0];
        assert_eq!(binding.prediction_kind, PredictionKind::RegressionPoint);
        preflight_facet_calibration(
            &export.graph,
            &export.context,
            &export.members,
            "result",
            "model",
            &binding.binding_id,
        )
        .unwrap();
    }

    #[test]
    fn actual_schema2_view_uses_pre_conformal_anchor_not_changed_outcome() {
        let before = before();
        let after = after();
        let model = facet_calibration_view(&after.context.owner_proofs["model"])
            .unwrap()
            .unwrap();
        let outcome = facet_calibration_view(&after.context.owner_proofs["result"])
            .unwrap()
            .unwrap();
        assert_eq!(
            model.pre_conformal_outcome_fingerprint,
            before.context.owner_proofs["result"].identity().fingerprint
        );
        assert_ne!(
            model.pre_conformal_outcome_fingerprint,
            after.context.owner_proofs["result"].identity().fingerprint
        );
        assert_eq!(
            model.calibration_fingerprint,
            outcome.calibration_fingerprint
        );
        assert_eq!(model.bundle_reference.schema_version, 2);
        assert_eq!(
            model.calibration_replay_outcome_fingerprint,
            TrainingReplayOutcome::from_json(REPLAY)
                .unwrap()
                .outcome_fingerprint
        );
        assert_eq!(state_bytes(&before), state_bytes(&after));
        assert_eq!(
            before.members["dagml/score.json"],
            after.members["dagml/score.json"]
        );
    }

    #[test]
    fn explicit_owner_calibrate_returns_complete_matching_set_without_extra_state_member() {
        let before = before();
        let replay = TrainingReplayOutcome::from_json(REPLAY).unwrap();
        let relations: SampleRelationSet = serde_json::from_str(CAL_RELATIONS).unwrap();
        let expected = after();
        let calibration = facet_calibration_view(&expected.context.owner_proofs["model"])
            .unwrap()
            .unwrap();
        let mut request = calibration_request(&replay, &relations, calibration.binding_id);
        let original: PortablePredictorPackage =
            PortablePredictorPackage::from_json(std::str::from_utf8(AFTER_PACKAGE).unwrap())
                .unwrap();
        request.coverages = original
            .conformal_calibration
            .as_ref()
            .unwrap()
            .coverages
            .clone();
        request.small_sample_policy = original
            .conformal_calibration
            .as_ref()
            .unwrap()
            .small_sample_policy;
        let actual = calibrate_training_artifact_facets(
            &before.graph,
            &before.context,
            &before.members,
            "result",
            "model",
            request,
        )
        .unwrap();
        actual.validate().unwrap();
        assert_eq!(actual.graph, expected.graph);
        assert_eq!(
            actual.context.owner_proofs["model"].identity(),
            expected.context.owner_proofs["model"].identity()
        );
        assert_eq!(state_bytes(&actual), state_bytes(&before));
        assert_eq!(
            actual.members.keys().collect::<Vec<_>>(),
            before.members.keys().collect::<Vec<_>>()
        );
        assert_eq!(
            actual.members["dagml/score.json"],
            before.members["dagml/score.json"]
        );
        assert!(matches!(
            calibrate_training_artifact_facets(
                &actual.graph,
                &actual.context,
                &actual.members,
                "result",
                "model",
                calibration_request(&replay, &relations, calibration.binding_id)
            ),
            Err(FacetRefusal::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn calibrated_owner_refuses_schema_binding_target_replay_and_post_anchor_contradictions() {
        let original =
            PortablePredictorPackage::from_json(std::str::from_utf8(AFTER_PACKAGE).unwrap())
                .unwrap();
        for mutation in 0..6 {
            let mut package = original.clone();
            match mutation {
                0 => {
                    package
                        .conformal_calibration
                        .as_mut()
                        .unwrap()
                        .schema_version = 1
                }
                1 => {
                    package.conformal_calibration.as_mut().unwrap().binding_id =
                        "binding:absent".into()
                }
                2 => {
                    package.conformal_calibration.as_mut().unwrap().target_names =
                        vec!["wrong-target".into()]
                }
                3 => {
                    package.training_outcome.pre_conformal_outcome_fingerprint =
                        Some(package.training_outcome.outcome_fingerprint.clone())
                }
                4 => {
                    package
                        .conformal_calibration
                        .as_mut()
                        .unwrap()
                        .context
                        .source_training_outcome_fingerprint =
                        package.training_outcome.outcome_fingerprint.clone()
                }
                5 => {
                    package
                        .conformal_calibration_replay
                        .as_mut()
                        .unwrap()
                        .source_training_outcome
                        .outcome_fingerprint = package.training_outcome.outcome_fingerprint.clone()
                }
                _ => unreachable!(),
            }
            // Recompute consistency hashes to challenge actual logical owner
            // closure checks, not merely the outer stored self-fingerprint.
            let calibration = package.conformal_calibration.as_mut().unwrap();
            calibration.context.context_fingerprint =
                calibration.context.compute_fingerprint().unwrap();
            calibration.calibration_fingerprint = calibration.compute_fingerprint().unwrap();
            package
                .execution_bundle
                .conformal_calibration
                .as_mut()
                .unwrap()
                .calibration_fingerprint = calibration.calibration_fingerprint.clone();
            package.package_fingerprint = package.compute_fingerprint().unwrap();
            assert!(matches!(
                validate_owner_facet_bytes(
                    &FacetContentType::new(FACET_PREDICTOR_PACKAGE, 2),
                    &serde_json::to_vec(&package).unwrap()
                ),
                Err(FacetRefusal::OwnerValidation { .. })
            ));
        }
    }

    #[test]
    fn calibrated_package_cannot_mix_pre_calibration_evidence_or_foreign_score() {
        for id in ["result", "score"] {
            let mut post = after();
            let old = if id == "result" { before() } else { source() };
            let old_artifact = old
                .graph
                .artifacts
                .iter()
                .find(|a| a.id == id)
                .unwrap()
                .clone();
            let artifact = post
                .graph
                .artifacts
                .iter_mut()
                .find(|a| a.id == id)
                .unwrap();
            *artifact = old_artifact;
            if let ArtifactFacetStorage::Embedded { root_member, .. } = &artifact.storage {
                post.members
                    .insert(root_member.clone(), old.members[root_member].clone());
            }
            post.context
                .owner_proofs
                .insert(id.into(), old.context.owner_proofs[id].clone());
            assert!(matches!(
                post.validate(),
                Err(FacetRefusal::Mismatch { .. })
            ));
        }
    }

    #[test]
    fn calibration_rejects_wrong_binding_and_reordered_truth_id_join() {
        let before = before();
        let replay = TrainingReplayOutcome::from_json(REPLAY).unwrap();
        let relations: SampleRelationSet = serde_json::from_str(CAL_RELATIONS).unwrap();
        let binding = &replay.outputs[0].binding.binding_id;
        for mutation in 0..3 {
            let mut request = calibration_request(&replay, &relations, binding);
            match mutation {
                0 => request.binding_id = "binding:absent",
                1 => request.truth.sample_ids.reverse(),
                2 => {
                    request.truth.sample_ids.reverse();
                    request.truth.values.reverse();
                }
                _ => unreachable!(),
            }
            let result = calibrate_training_artifact_facets(
                &before.graph,
                &before.context,
                &before.members,
                "result",
                "model",
                request,
            );
            if mutation == 0 {
                assert!(matches!(
                    result,
                    Err(FacetRefusal::Mismatch { detail })
                        if detail == "requested calibration binding is not a validated package output binding"
                ));
            } else {
                assert!(matches!(result, Err(FacetRefusal::OwnerValidation { .. })));
            }
        }
    }

    #[test]
    fn owner_derived_context_refuses_physical_and_origin_training_overlap() {
        let before = before();
        let source = TrainingOutcome::from_json(std::str::from_utf8(BEFORE).unwrap()).unwrap();
        let training_id = source
            .training_influence
            .entries
            .iter()
            .flat_map(|entry| &entry.physical_sample_ids)
            .next()
            .unwrap()
            .clone();
        for physical_overlap in [false, true] {
            // Negative identity mutations of genuine replay evidence, never
            // fabricated predictions or a positive native execution claim.
            let mut replay = TrainingReplayOutcome::from_json(REPLAY).unwrap();
            let mut relations: SampleRelationSet = serde_json::from_str(CAL_RELATIONS).unwrap();
            let binding = replay.outputs[0].binding.binding_id.clone();
            let mut request = calibration_request(&replay, &relations, &binding);
            let mut truth = request.truth;
            let first = truth.sample_ids[0].clone();
            for relation in &mut relations.records {
                if relation.sample_id == first {
                    if physical_overlap {
                        relation.sample_id = training_id.clone();
                    } else {
                        relation.origin_sample_id = Some(training_id.clone());
                    }
                }
            }
            if physical_overlap {
                let point = &mut replay.outputs[0].predictions[0];
                point.sample_ids[0] = training_id.clone();
                truth.sample_ids[0] = training_id.clone();
                let mut rows = point
                    .sample_ids
                    .iter()
                    .cloned()
                    .zip(point.values.clone())
                    .zip(truth.values.clone())
                    .collect::<Vec<_>>();
                rows.sort_by(|left, right| left.0 .0.cmp(&right.0 .0));
                point.sample_ids = rows.iter().map(|row| row.0 .0.clone()).collect();
                point.values = rows.iter().map(|row| row.0 .1.clone()).collect();
                truth.sample_ids = point.sample_ids.clone();
                truth.values = rows.into_iter().map(|row| row.1).collect();
            }
            let relation_fingerprint = relations.fingerprint().unwrap();
            for identity in &mut replay.input_data_identities {
                identity.relation_fingerprint = relation_fingerprint.clone();
                identity.identity_fingerprint = identity.compute_fingerprint().unwrap();
            }
            replay.outcome_fingerprint = replay.compute_fingerprint().unwrap();
            let context = derive_attached_conformal_calibration_context(
                &source, &replay, &binding, &relations,
            )
            .unwrap();
            let ids = if physical_overlap {
                &context.calibration_cohort.physical_sample_ids
            } else {
                &context.calibration_cohort.origin_sample_ids
            };
            assert!(ids.contains(&training_id));
            request = calibration_request(&replay, &relations, &binding);
            request.truth = truth;
            assert!(matches!(
                calibrate_training_artifact_facets(
                    &before.graph,
                    &before.context,
                    &before.members,
                    "result",
                    "model",
                    request
                ),
                Err(FacetRefusal::OwnerValidation { .. })
            ));
        }
    }
    #[test]
    fn calibration_preflight_and_action_share_static_binding_and_order() {
        let before = before();
        let original_graph = before.graph.clone();
        let original_members = before.members.clone();
        let states = state_bytes(&before);
        let detail = "requested calibration binding is not a validated package output binding";
        let check = |result: FacetResult<()>| match result {
            Err(FacetRefusal::Mismatch { detail: actual }) => assert_eq!(actual, detail),
            other => panic!("unexpected refusal: {other:?}"),
        };
        check(preflight_facet_calibration(
            &before.graph,
            &before.context,
            &before.members,
            "result",
            "model",
            "unknown",
        ));
        let replay = TrainingReplayOutcome::from_json(REPLAY).unwrap();
        let relations: SampleRelationSet = serde_json::from_str(CAL_RELATIONS).unwrap();
        let action = calibrate_training_artifact_facets(
            &before.graph,
            &before.context,
            &before.members,
            "result",
            "model",
            calibration_request(&replay, &relations, "unknown"),
        );
        check(action.map(|_| ()));
        assert!(matches!(
            preflight_facet_calibration(
                &before.graph,
                &before.context,
                &before.members,
                "missing",
                "model",
                "unknown"
            ),
            Err(FacetRefusal::InvalidStructure { .. })
        ));
        let mut missing = before.members.clone();
        missing.remove("dagml/model.json");
        assert!(matches!(
            preflight_facet_calibration(
                &before.graph,
                &before.context,
                &missing,
                "missing",
                "model",
                "unknown"
            ),
            Err(FacetRefusal::Mismatch { .. }) | Err(FacetRefusal::InvalidStructure { .. })
        ));
        assert_eq!(before.graph, original_graph);
        assert_eq!(before.members, original_members);
        assert_eq!(state_bytes(&before), states);
        let mut unresolved = self::before();
        unresolved.context.owner_proofs.remove("model");
        assert!(matches!(
            preflight_facet_calibration(
                &unresolved.graph,
                &unresolved.context,
                &unresolved.members,
                "result",
                "model",
                "unknown"
            ),
            Err(FacetRefusal::UnresolvedOwnerContent { .. })
        ));
        let mut inconsistent = after();
        let old = self::before();
        let result = inconsistent
            .graph
            .artifacts
            .iter_mut()
            .find(|a| a.id == "result")
            .unwrap();
        *result = old
            .graph
            .artifacts
            .iter()
            .find(|a| a.id == "result")
            .unwrap()
            .clone();
        inconsistent.members.insert(
            "dagml/result.json".into(),
            old.members["dagml/result.json"].clone(),
        );
        inconsistent
            .context
            .owner_proofs
            .insert("result".into(), old.context.owner_proofs["result"].clone());
        assert!(matches!(
            preflight_facet_calibration(
                &inconsistent.graph,
                &inconsistent.context,
                &inconsistent.members,
                "result",
                "model",
                "unknown"
            ),
            Err(FacetRefusal::Mismatch { .. })
        ));
        let calibrated = after();
        assert!(matches!(
            preflight_facet_calibration(
                &calibrated.graph,
                &calibrated.context,
                &calibrated.members,
                "result",
                "model",
                "unknown"
            ),
            Err(FacetRefusal::UnsupportedOperation { .. })
        ));
    }
}
