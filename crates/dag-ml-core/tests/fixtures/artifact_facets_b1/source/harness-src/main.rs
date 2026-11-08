//! B1 calibration fixture generator (staging only; Core-free).
//!
//! Composes only published owner APIs: IO normalization/package projection, the DAG-ML training
//! operation, the DAG-ML callback-free native Methods PREDICT replay and
//! `calibrate_attached_training_replay_with_derived_context`. No numerics and no fingerprint are
//! implemented or edited here; every fingerprint/quantile is computed by the owner crates.

mod host_provider;

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::path::{Path, PathBuf};

use dag_ml_core::{
    calibrate_attached_training_replay_with_derived_context, data_binding_requirement_key,
    execute_loaded_methods_predictor_replay, execute_training, register_methods_estimator_controllers,
    ArtifactLoadMode, BundleId, ConformalCalibrationTruth, ConformalMultiTargetPolicy,
    ConformalSmallSamplePolicy, FittedArtifactMode, InMemoryArtifactStore,
    MethodsPlsController, MethodsPlsDataset, MethodsPlsMatrix, MethodsPortablePredictorReplayInput,
    MethodsRuntime, Phase, PortablePredictorPackage, PredictCohort, PredictCohortRole, RunId,
    RuntimeControllerRegistry, SampleId, TrainingExecutionInput, TrainingInfluenceManifest,
    TrainingOutcome, TrainingReplayOutcome, TrainingReplayRequest,
};
use host_provider::{canonical_pls_training_request, CanonicalPlsProfile, DatasetPackageMethodsProvider};
use nirs4all_io::core::public_dataset::{matrix_dataset_package, normalize_dataset};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

const COVERAGES: [f64; 3] = [0.8, 0.9, 0.95];

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write(path: impl AsRef<Path>, bytes: impl AsRef<[u8]>) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec(value)?;
    write(path, &bytes)?;
    Ok(bytes)
}

fn raw_state(package: &PortablePredictorPackage) -> BTreeMap<String, Vec<u8>> {
    package
        .execution_bundle
        .raw_artifact_payloads
        .iter()
        .map(|(id, bytes)| (id.to_string(), bytes.clone()))
        .collect()
}

fn raw_state_hashes(state: &BTreeMap<String, Vec<u8>>) -> BTreeMap<String, Value> {
    state
        .iter()
        .map(|(id, bytes)| (id.clone(), json!({"bytes": bytes.len(), "sha256": sha256(bytes)})))
        .collect()
}

fn native_info(state: &BTreeMap<String, Vec<u8>>) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    for (id, bytes) in state {
        let info = n4m::inspect_n4mm(bytes)?;
        if info.format_version != 3 {
            return Err(format!("artifact `{id}` is N4MM format {}, expected 3", info.format_version).into());
        }
        out.push(json!({"artifact_id": id, "format_version": info.format_version,
            "writer_abi": info.writer_abi, "n_features": info.n_features,
            "n_targets": info.n_targets, "raw_sha256": sha256(bytes)}));
    }
    Ok(out)
}

fn load_record(path: &Path) -> Result<(Vec<u8>, Value, Vec<u8>)> {
    let raw = std::fs::read(path)?;
    let normalized = normalize_dataset(&serde_json::from_slice(&raw)?)?;
    let normalized_bytes = serde_json::to_vec(&normalized)?;
    Ok((raw, normalized, normalized_bytes))
}

fn library_sha(library: &Path) -> Result<String> {
    Ok(sha256(&std::fs::read(library)?))
}

/// Truth from the IO owner's target block for exactly the replay point-prediction identities.
fn calibration_truth(
    package: &nirs4all_io::core::materialize::DatasetPackage,
    point_ids: &[SampleId],
    expected_x: &MethodsPlsMatrix,
) -> Result<ConformalCalibrationTruth> {
    use dag_ml_data::{CoordinatorDataMaterializationRequest, DataView, SampleId as IoSampleId, SourceId};
    use dag_ml_data_provider::DagMlDataProvider;
    let provider = nirs4all_io_dagml::PackageProvider::from_package_source(package, "spectra")?;
    let envelope = provider.envelope();
    let source = SourceId::new("spectra")?;
    let handle = provider.materialize(&CoordinatorDataMaterializationRequest {
        run_id: "run:b1.calibration.truth".into(), node_id: "model:pls".into(),
        input_name: "x".into(), phase: "PREDICT".into(), variant_id: None, fold_id: None,
        request_id: "io:b1-calibration-truth".into(),
        schema_fingerprint: envelope.schema_fingerprint.clone(),
        plan_fingerprint: envelope.plan_fingerprint.clone(),
        relation_fingerprint: envelope.relation_fingerprint.clone(),
        output_representation: envelope.plan.output_representation.clone(),
        source_ids: vec![source.clone()], require_relations: true, predict_cohort: None,
    })?;
    let ids = point_ids.iter().map(|id| IoSampleId::new(id.as_str())).collect::<std::result::Result<Vec<_>, _>>()?;
    let view = provider.make_view(handle.handle.handle, &DataView {
        sample_ids: Some(ids.clone()), partition: None, fold_id: None, source_ids: Some(vec![source]),
        columns: None, include_augmented: false, include_excluded: false, branch_view: None,
        extra: Default::default(),
    })?;
    let feature = provider.feature_block_f64(view.handle.handle)?;
    if feature.sample_ids != ids || feature.values != expected_x.values {
        return Err("IO view features differ from the identity-keyed replay input".into());
    }
    let [target_id] = provider.target_ids() else {
        return Err("fixture expects one target id".into());
    };
    let block = provider.target_block(view.handle.handle, target_id)?;
    if block.sample_ids != ids {
        return Err("IO target rows differ from view identities".into());
    }
    let values = block.values.iter().map(|v| v.as_f64().map(|x| vec![x]).ok_or("non-numeric truth"))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    provider.release(view.handle.handle);
    provider.release(handle.handle.handle);
    Ok(ConformalCalibrationTruth { sample_ids: point_ids.to_vec(), values })
}

fn produce(inputs: &Path, out: &Path, library: &Path) -> Result<()> {
    let lib_before = library_sha(library)?;
    std::fs::create_dir_all(out)?;
    let mut facts = serde_json::Map::new();

    // ---- Source/training outcome (genuine native N4MM3 two-fold PLS) -------------------------
    let (src_raw, src_norm, src_norm_bytes) = load_record(&inputs.join("source_dataset.json"))?;
    write(out.join("source_dataset.normalized.json"), &src_norm_bytes)?;
    let src_pkg = matrix_dataset_package(&src_norm, "spectra")?;
    let src_provider = DatasetPackageMethodsProvider::new(&src_pkg, "spectra")?;
    let request = canonical_pls_training_request(&src_provider, CanonicalPlsProfile::Raw)?;
    request.validate()?;
    let projection = request.project()?;
    let influence = TrainingInfluenceManifest::derive_for_projection(&projection, &request, src_provider.relations())?;
    let runtime = MethodsRuntime::configure(library)?;
    let mut controllers = RuntimeControllerRegistry::new();
    register_methods_estimator_controllers(&mut controllers, runtime.clone())?;
    controllers.register(Box::new(MethodsPlsController::new(runtime.clone())))?;
    let mut store = InMemoryArtifactStore::new();
    let mut outcome = execute_training(TrainingExecutionInput {
        request: &request,
        outcome_id: "outcome:b1.calibration.fixture".into(),
        run_id: RunId::new("run:b1.calibration.fixture")?,
        bundle_id: BundleId::new("bundle:b1.calibration.fixture")?,
        controllers: &controllers,
        data_provider: &src_provider,
        relations: src_provider.relations(),
        training_influence: &influence,
        artifact_store: &mut store,
        warnings: Vec::new(),
        diagnostics: BTreeMap::new(),
    })?;
    outcome.validate()?;
    let package_before = outcome.to_portable_predictor_package(
        "package:b1.calibration.fixture", FittedArtifactMode::PortableRequired, ArtifactLoadMode::NativePortable,
    )?;
    package_before.validate()?;
    let outcome_before_bytes = write_json(out.join("training_outcome.before.json"), &outcome)?;
    let ppc_before_bytes = write_json(out.join("ppc.before.json"), &package_before)?;
    let fp_before = outcome.outcome_fingerprint.clone();
    if package_before.training_outcome.outcome_fingerprint != fp_before
        || package_before.training_outcome.pre_conformal_outcome_fingerprint.is_some()
    {
        return Err("PPC-before does not anchor the uncalibrated source outcome".into());
    }
    let state_before = raw_state(&package_before);
    let native_before = native_info(&state_before)?;
    // Strict raw-byte TCV1 reload of the exact written bytes.
    TrainingOutcome::from_json(std::str::from_utf8(&outcome_before_bytes)?)?;
    PortablePredictorPackage::from_json(std::str::from_utf8(&ppc_before_bytes)?)?;
    for (id, bytes) in &state_before {
        write(out.join("raw_state_before").join(format!("{}.n4mm", id.replace([':', '/'], "_"))), bytes)?;
    }

    // ---- Held-out target-bound calibration cohort ---------------------------------------------
    let (cal_raw, cal_norm, cal_norm_bytes) = load_record(&inputs.join("calibration_dataset.json"))?;
    write(out.join("calibration_dataset.normalized.json"), &cal_norm_bytes)?;
    let cal_pkg = matrix_dataset_package(&cal_norm, "spectra")?;
    let cal_provider = DatasetPackageMethodsProvider::new(&cal_pkg, "spectra")?;
    let cal_cohort = nirs4all_io_dagml::methods_pls_predict_cohort(&cal_pkg.to_assembled(), "test", "spectra")?;
    let output = package_before.output_bindings.first().ok_or("missing package output binding")?;
    let binding_id = output.binding_id.clone();
    let target_names = output.target_names.clone();
    let predict_cohort = PredictCohort::from_relations(
        PredictCohortRole::ExternalTest,
        cal_provider.relations().clone(),
        target_names.clone(),
        cal_cohort.data_content_fingerprint.clone(),
        cal_provider.external_envelope().target_content_fingerprint.clone(),
    )?;
    let mut envelope = src_provider.external_envelope().clone();
    envelope.schema_version = 2;
    envelope.data_content_fingerprint = Some(cal_cohort.data_content_fingerprint.clone());
    envelope.target_content_fingerprint = cal_provider.external_envelope().target_content_fingerprint.clone();
    envelope.predict_cohort = Some(predict_cohort.clone());
    envelope.validate()?; // includes owner CV-disjointness proof for ExternalTest
    let requirement = package_before.execution_bundle.data_requirements.first().ok_or("missing data requirement")?;
    let key = requirement.key();
    if key != data_binding_requirement_key(&requirement.node_id, &requirement.input_name) {
        return Err("requirement key mismatch".into());
    }
    let dataset = MethodsPlsDataset {
        sample_ids: cal_cohort.sample_ids.iter().map(|id| SampleId::new(id)).collect::<dag_ml_core::Result<_>>()?,
        x: MethodsPlsMatrix { values: cal_cohort.values.clone(), rows: cal_cohort.rows, cols: cal_cohort.cols },
        y: None, y_validity_masks: None, target_names: target_names.clone(),
    };
    dataset.validate("B1 calibration cohort", false)?;
    let mut replay_request = TrainingReplayRequest {
        schema_version: 1, request_id: "predict:b1.calibration.fixture".into(),
        source_outcome_fingerprint: package_before.training_outcome.outcome_fingerprint.clone(),
        phase: Phase::Predict, data_envelope_keys: vec![key.clone()],
        output_binding_ids: vec![binding_id.clone()], request_fingerprint: String::new(),
    };
    replay_request.request_fingerprint = replay_request.compute_fingerprint()?;
    replay_request.validate()?;
    let envelopes = BTreeMap::from([(key.clone(), envelope.clone())]);
    write_json(out.join("replay_request.json"), &replay_request)?;
    write_json(out.join("replay_envelope.json"), &envelopes)?;
    let replay = execute_loaded_methods_predictor_replay(MethodsPortablePredictorReplayInput {
        package: &package_before, request: &replay_request, data_envelopes: &envelopes,
        methods_inputs: &BTreeMap::from([(key.clone(), dataset.clone())]),
        runtime: runtime.clone(),
        outcome_id: "outcome:b1.calibration.fixture.predict".into(),
        run_id: RunId::new("run:b1.calibration.fixture.predict")?, warnings: vec![], diagnostics: BTreeMap::new(),
    })?;
    replay.validate_against_package(&package_before, &replay_request)?;
    let replay_bytes = write_json(out.join("calibration_replay.json"), &replay)?;
    TrainingReplayOutcome::from_json(std::str::from_utf8(&replay_bytes)?)?;
    let output_block = replay.outputs.iter().find(|o| o.binding.binding_id == binding_id).ok_or("replay output missing")?;
    let [point] = output_block.predictions.as_slice() else { return Err("expected one point block".into()) };
    let truth = calibration_truth(&cal_pkg, &point.sample_ids, &dataset.x)?;
    write_json(out.join("calibration_truth.json"), &json!({
        "target_names": output_block.binding.target_names, "sample_ids": truth.sample_ids, "values": truth.values}))?;
    write_json(out.join("calibration_relations.json"), cal_provider.relations())?;

    // ---- Disjointness evidence (computed from the owner-held manifests) -------------------------
    let influence_ids: BTreeSet<String> = outcome.training_influence.entries.iter()
        .flat_map(|e| e.physical_sample_ids.iter().chain(e.origin_sample_ids.iter()))
        .map(|id| id.to_string()).collect();
    let foldset_ids: BTreeSet<String> = outcome.effective_plan.fold_set.as_ref().ok_or("no fold set")?
        .sample_ids.iter().map(|id| id.to_string()).collect();
    let cal_physical: BTreeSet<String> = predict_cohort.physical_sample_ids.iter().map(|id| id.to_string()).collect();
    let cal_origin: BTreeSet<String> = predict_cohort.origin_sample_ids.iter().map(|id| id.to_string()).collect();
    let src_relation_ids: BTreeSet<String> = src_provider.relations().records.iter()
        .flat_map(|r| [Some(r.sample_id.to_string()), r.origin_sample_id.as_ref().map(|i| i.to_string())])
        .flatten().collect();
    let inter = |a: &BTreeSet<String>, b: &BTreeSet<String>| a.intersection(b).cloned().collect::<Vec<_>>();
    let disjoint = json!({
        "calibration_physical_count": cal_physical.len(), "calibration_origin_count": cal_origin.len(),
        "source_training_influence_id_count": influence_ids.len(), "source_foldset_sample_count": foldset_ids.len(),
        "source_relation_id_count": src_relation_ids.len(),
        "physical_x_influence": inter(&cal_physical, &influence_ids),
        "origin_x_influence": inter(&cal_origin, &influence_ids),
        "physical_x_foldset": inter(&cal_physical, &foldset_ids),
        "origin_x_foldset": inter(&cal_origin, &foldset_ids),
        "physical_x_source_relations": inter(&cal_physical, &src_relation_ids),
        "origin_x_source_relations": inter(&cal_origin, &src_relation_ids),
        "relation_fingerprint_calibration": predict_cohort.relation_fingerprint,
        "relation_fingerprint_source_development": outcome.training_influence.relation_fingerprint,
    });
    for k in ["physical_x_influence","origin_x_influence","physical_x_foldset","origin_x_foldset","physical_x_source_relations","origin_x_source_relations"] {
        if !disjoint[k].as_array().ok_or("bad")?.is_empty() { return Err(format!("overlap in {k}").into()); }
    }
    if predict_cohort.relation_fingerprint == outcome.training_influence.relation_fingerprint {
        return Err("calibration relation authority equals development relations".into());
    }
    write_json(out.join("disjointness.json"), &disjoint)?;

    // ---- Owner calibration (existing code computes everything) ------------------------------------
    let calibration = calibrate_attached_training_replay_with_derived_context(
        &mut outcome, &replay, &binding_id, cal_provider.relations(), truth.clone(),
        COVERAGES.to_vec(), ConformalMultiTargetPolicy::Marginal, ConformalSmallSamplePolicy::Error,
    )?;
    outcome.validate()?;
    let fp_after = outcome.outcome_fingerprint.clone();
    if fp_after == fp_before { return Err("calibration did not change the source outcome fingerprint".into()); }
    let package_after = outcome.to_portable_predictor_package(
        "package:b1.calibration.fixture", FittedArtifactMode::PortableRequired, ArtifactLoadMode::NativePortable,
    )?;
    package_after.validate()?;
    let outcome_after_bytes = write_json(out.join("training_outcome.after.json"), &outcome)?;
    let ppc_after_bytes = write_json(out.join("ppc.after.json"), &package_after)?;
    write_json(out.join("conformal_calibration.json"), &calibration)?;
    TrainingOutcome::from_json(std::str::from_utf8(&outcome_after_bytes)?)?;
    PortablePredictorPackage::from_json(std::str::from_utf8(&ppc_after_bytes)?)?;
    let anchor = &package_after.training_outcome;
    if anchor.outcome_fingerprint != fp_after
        || anchor.pre_conformal_outcome_fingerprint.as_deref() != Some(fp_before.as_str())
    {
        return Err("PPC-after anchors do not match (after, before-calibration source)".into());
    }
    // The pre-calibration replay is still valid against the AFTER outcome's retained pre-conformal source.
    let replay_after = outcome.conformal_calibration_replay.as_ref().ok_or("replay not attached")?;
    if replay_after != &replay { return Err("attached replay differs from the executed replay".into()); }
    let state_after = raw_state(&package_after);
    if state_before != state_after { return Err("raw native state bytes changed across calibration".into()); }
    let native_after = native_info(&state_after)?;
    // Second, independent native replay of the CALIBRATED package on the same cohort.
    let mut request_after = TrainingReplayRequest {
        schema_version: 1, request_id: "predict:b1.calibration.fixture.after".into(),
        source_outcome_fingerprint: package_after.training_outcome.outcome_fingerprint.clone(),
        phase: Phase::Predict, data_envelope_keys: vec![key.clone()],
        output_binding_ids: vec![binding_id.clone()], request_fingerprint: String::new(),
    };
    request_after.request_fingerprint = request_after.compute_fingerprint()?;
    let replay_calibrated = execute_loaded_methods_predictor_replay(MethodsPortablePredictorReplayInput {
        package: &package_after, request: &request_after, data_envelopes: &envelopes,
        methods_inputs: &BTreeMap::from([(key.clone(), dataset.clone())]), runtime: runtime.clone(),
        outcome_id: "outcome:b1.calibration.fixture.predict.after".into(),
        run_id: RunId::new("run:b1.calibration.fixture.predict.after")?, warnings: vec![], diagnostics: BTreeMap::new(),
    });
    let calibrated_replay_status = match replay_calibrated {
        Ok(r) => {
            r.validate_against_package(&package_after, &request_after)?;
            write_json(out.join("calibrated_package_replay.json"), &r)?;
            let same = r.outputs.iter().map(|o| &o.predictions).collect::<Vec<_>>()
                == replay.outputs.iter().map(|o| &o.predictions).collect::<Vec<_>>();
            json!({"status": "passed", "point_predictions_identical_to_pre_calibration_replay": same,
                   "conformal_interval_blocks": r.conformal_intervals.len()})
        }
        Err(e) => json!({"status": "refused", "error": e.to_string()}),
    };
    let lib_after = library_sha(library)?;
    let predictions: Vec<Value> = point.sample_ids.iter().enumerate().map(|(i, id)| {
        json!({"sample_id": id, "predicted": point.values.get(i), "truth": truth.values[i]})
    }).collect();
    write_json(out.join("native_predictions_vs_truth.json"), &json!({
        "target_names": output_block.binding.target_names, "rows": predictions}))?;

    facts.insert("scope".into(), json!("controlled synthetic calibration fixture; inner schema2 ConformalCalibration only; no real-measurement or coverage-generalization claim"));
    facts.insert("source_input_sha256".into(), json!({"source_dataset.json": sha256(&src_raw), "calibration_dataset.json": sha256(&cal_raw)}));
    facts.insert("normalized_sha256".into(), json!({"source": sha256(&src_norm_bytes), "calibration": sha256(&cal_norm_bytes)}));
    facts.insert("library_sha256".into(), json!({"before": lib_before, "after": lib_after}));
    facts.insert("source_outcome_fingerprint_before_calibration".into(), json!(fp_before));
    facts.insert("source_outcome_fingerprint_after_calibration".into(), json!(fp_after));
    facts.insert("ppc_before_anchor".into(), serde_json::to_value(&package_before.training_outcome)?);
    facts.insert("ppc_after_anchor".into(), serde_json::to_value(&package_after.training_outcome)?);
    facts.insert("ppc_before_package_fingerprint".into(), json!(package_before.package_fingerprint));
    facts.insert("ppc_after_package_fingerprint".into(), json!(package_after.package_fingerprint));
    facts.insert("native_model_info_before".into(), json!(native_before));
    facts.insert("native_model_info_after".into(), json!(native_after));
    facts.insert("raw_state_hashes_before".into(), json!(raw_state_hashes(&state_before)));
    facts.insert("raw_state_hashes_after".into(), json!(raw_state_hashes(&state_after)));
    facts.insert("raw_state_identical_before_after".into(), json!(state_before == state_after));
    facts.insert("calibration".into(), serde_json::to_value(&calibration)?);
    facts.insert("calibration_replay_outcome_fingerprint".into(), json!(replay.outcome_fingerprint));
    facts.insert("calibration_replay_input_identities".into(), serde_json::to_value(&replay.input_data_identities)?);
    facts.insert("calibrated_package_replay".into(), calibrated_replay_status);
    facts.insert("coverages".into(), json!(COVERAGES));
    facts.insert("calibration_sample_count".into(), json!(truth.sample_ids.len()));
    facts.insert("binding_id".into(), json!(binding_id));
    facts.insert("target_names".into(), json!(target_names));
    facts.insert("disjointness".into(), disjoint);
    facts.insert("file_sha256".into(), {
        let mut m = serde_json::Map::new();
        for entry in std::fs::read_dir(out)? {
            let p = entry?.path();
            if p.is_file() { m.insert(p.file_name().unwrap().to_string_lossy().into(), json!(sha256(&std::fs::read(&p)?))); }
        }
        Value::Object(m)
    });
    write(out.join("producer_facts.json"), serde_json::to_vec_pretty(&Value::Object(facts))?)?;
    src_provider.release_all();
    cal_provider.release_all();
    println!("B1 calibration fixture produced: before={fp_before} after={fp_after}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 || args[1] != "produce" {
        return Err("Usage: b1-calibration-fixture produce INPUT_DIR OUTPUT_DIR METHODS_LIBRARY".into());
    }
    produce(&PathBuf::from(&args[2]), &PathBuf::from(&args[3]), &PathBuf::from(&args[4]))
}
