//! End-to-end no-CV REFIT and independent PREDICT through C controller vtables.

use std::ffi::c_void;
use std::slice;

use dag_ml_capi::{
    dagml_initial_full_refit_execute_json, dagml_initial_full_refit_predict_json,
    dagml_owned_bytes_free, dagml_string_free, DagMlBytesView, DagMlControllerBinding,
    DagMlControllerVTable, DagMlInitialFullRefitExecuteRequest,
    DagMlInitialFullRefitPredictRequest, DagMlOwnedBytes, DagMlStatusCode, DagMlString,
    DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION,
};
use dag_ml_core::{
    build_execution_plan, CampaignSpec, ControllerManifest, ControllerRegistry, GraphSpec,
    InitialFullRefitPackage, PredictCohort, PredictCohortRole, SampleRelationSet,
};
use serde_json::{json, Value};

const PACKAGE: &str =
    include_str!("../../dag-ml-core/tests/fixtures/initial_full_refit/package.json");
const REFIT_RESULT: &str =
    include_str!("../../dag-ml-core/tests/fixtures/initial_full_refit/refit_node_result.json");

fn view(bytes: &[u8]) -> DagMlBytesView {
    DagMlBytesView {
        ptr: bytes.as_ptr(),
        len: bytes.len(),
    }
}

fn owned_json(value: DagMlOwnedBytes) -> Value {
    let json =
        serde_json::from_slice(unsafe { slice::from_raw_parts(value.ptr, value.len) }).unwrap();
    unsafe { dagml_owned_bytes_free(value) };
    json
}

fn error_message(error: DagMlString) -> String {
    let text = if error.ptr.is_null() {
        String::new()
    } else {
        String::from_utf8_lossy(unsafe { slice::from_raw_parts(error.ptr.cast::<u8>(), error.len) })
            .into_owned()
    };
    unsafe { dagml_string_free(error) };
    text
}

#[derive(Default)]
struct ControllerState {
    calls: usize,
}

unsafe extern "C" fn invoke(
    user_data: *mut c_void,
    task_json: DagMlBytesView,
    out_json: *mut DagMlOwnedBytes,
) -> DagMlStatusCode {
    if user_data.is_null() || task_json.ptr.is_null() || out_json.is_null() {
        return DagMlStatusCode::INVALID_ARGUMENT;
    }
    let state = &mut *user_data.cast::<ControllerState>();
    state.calls += 1;
    let task: Value =
        match serde_json::from_slice(slice::from_raw_parts(task_json.ptr, task_json.len)) {
            Ok(task) => task,
            Err(_) => return DagMlStatusCode::VALIDATION_ERROR,
        };
    let mut result: Value = serde_json::from_str(REFIT_RESULT).unwrap();
    let node = task["node_plan"]["node_id"].as_str().unwrap();
    result["lineage"]["run_id"] = task["run_id"].clone();
    result["lineage"]["seed"] = task["seed"].clone();
    result["lineage"]["record_id"] = json!(format!(
        "lineage:cabi.initial.{}:{node}",
        task["phase"].as_str().unwrap()
    ));
    result["predictions"][0]["prediction_id"] = json!(format!(
        "pred:cabi.initial.{}:{node}",
        task["phase"].as_str().unwrap()
    ));
    if node != "model:initial" {
        let old_artifact = "artifact:model:initial:refit";
        let new_artifact = format!("artifact:{node}:refit");
        result["node_id"] = json!(node);
        result["lineage"]["node_id"] = json!(node);
        result["artifacts"][0]["id"] = json!(new_artifact);
        result["lineage"]["artifact_refs"][0]["id"] = json!(new_artifact);
        result["artifact_handles"][&new_artifact] = result["artifact_handles"][old_artifact].take();
        result["artifact_handles"]
            .as_object_mut()
            .unwrap()
            .remove(old_artifact);
        result["predictions"][0]["producer_node"] = json!(node);
    }
    if task["phase"] == "PREDICT" {
        result["artifacts"] = json!([]);
        result["artifact_handles"] = json!({});
        result["lineage"]["artifact_refs"] = json!([]);
        result["lineage"]["phase"] = json!("PREDICT");
        result["lineage"]["run_id"] = task["run_id"].clone();
        result["lineage"]["seed"] = task["seed"].clone();
        result["lineage"]["record_id"] = json!(format!("lineage:cabi.initial.predict:{node}"));
        result["predictions"][0]["prediction_id"] =
            json!(format!("pred:cabi.initial.predict:{node}"));
        result["predictions"][0]["sample_ids"] = json!(["sample:heldout:1"]);
        result["predictions"][0]["values"] = json!([[7.0]]);
    }
    let mut bytes = serde_json::to_vec(&result).unwrap();
    *out_json = DagMlOwnedBytes {
        ptr: bytes.as_mut_ptr(),
        len: bytes.len(),
        capacity: bytes.capacity(),
    };
    std::mem::forget(bytes);
    DagMlStatusCode::OK
}

#[test]
fn c_abi_initial_full_refit_replays_two_independent_outputs() {
    let fixture = InitialFullRefitPackage::from_json(PACKAGE).unwrap();
    let mut graph = serde_json::to_value(&fixture.effective_plan.graph_plan.graph).unwrap();
    let mut second = graph["nodes"][0].clone();
    second["id"] = json!("model:second");
    graph["nodes"].as_array_mut().unwrap().push(second);
    let graph: GraphSpec = serde_json::from_value(graph).unwrap();
    let mut campaign = serde_json::to_value(&fixture.effective_plan.campaign).unwrap();
    let mut second_binding = campaign["data_bindings"]["model:initial"][0].clone();
    second_binding["node_id"] = json!("model:second");
    campaign["data_bindings"]["model:second"] = json!([second_binding]);
    let campaign: CampaignSpec = serde_json::from_value(campaign).unwrap();
    let mut registry = ControllerRegistry::new();
    for manifest in fixture.effective_plan.controller_manifests.values() {
        registry
            .register(ControllerManifest::clone(manifest))
            .unwrap();
    }
    let plan = build_execution_plan("plan:multi-output", graph, campaign, &registry).unwrap();
    let plan_json = serde_json::to_vec(&plan).unwrap();
    let envelope_json = serde_json::to_vec(&fixture.training_envelope).unwrap();
    let manifests_json = serde_json::to_vec(&registry.manifests().collect::<Vec<_>>()).unwrap();
    let ids_json = serde_json::to_vec(&fixture.training_sample_ids).unwrap();
    let mut state = ControllerState::default();
    let binding = DagMlControllerBinding {
        controller_id: view(b"controller:model.mock"),
        vtable: DagMlControllerVTable {
            abi_version: DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION,
            user_data: (&mut state as *mut ControllerState).cast(),
            clone_with: None,
            describe: None,
            fit: None,
            predict: None,
            invoke: Some(invoke),
            release_bytes: Some(release_bytes),
            release: None,
            destroy: None,
        },
    };
    let request = DagMlInitialFullRefitExecuteRequest {
        plan_json: view(&plan_json),
        envelope_json: view(&envelope_json),
        trusted_controllers_json: view(&manifests_json),
        training_sample_ids_json: view(&ids_json),
        package_id: view(b"package:multi-output"),
        run_id: view(b"run:multi-output.refit"),
        root_seed: 12345,
        controller_bindings: &binding,
        controller_binding_count: 1,
    };
    let mut out = DagMlOwnedBytes::default();
    let mut error = DagMlString::default();
    let status = unsafe { dagml_initial_full_refit_execute_json(&request, &mut out, &mut error) };
    assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(error));
    let capture = owned_json(out);
    let package_json = serde_json::to_vec(&capture["initial_full_refit_package"]).unwrap();
    let package =
        InitialFullRefitPackage::from_json(std::str::from_utf8(&package_json).unwrap()).unwrap();
    assert_eq!(package.outputs.len(), 2);
    assert_eq!(package.artifacts.len(), 2);
    assert_eq!(state.calls, 2);

    let heldout: SampleRelationSet = serde_json::from_value(json!({"records": [{
        "observation_id": "obs.H001", "sample_id": "sample:heldout:1",
        "target_id": "target:heldout:1", "group_id": "group:heldout",
        "origin_sample_id": null, "source_id": "nir", "is_augmented": false
    }]}))
    .unwrap();
    let cohort = PredictCohort::from_relations(
        PredictCohortRole::ExternalTest,
        heldout,
        vec!["y".into()],
        "a".repeat(64),
        Some("b".repeat(64)),
    )
    .unwrap();
    let envelope_json = serde_json::to_vec(&package.predict_envelope(cohort).unwrap()).unwrap();
    let output_ids = package
        .outputs
        .iter()
        .map(|output| output.output_id.clone())
        .collect::<Vec<_>>();
    let output_ids_json = serde_json::to_vec(&output_ids).unwrap();
    let handles = capture["node_results"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|result| result["artifact_handles"].as_object().unwrap())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<serde_json::Map<_, _>>();
    let handles_json = serde_json::to_vec(&handles).unwrap();
    let predict_request = DagMlInitialFullRefitPredictRequest {
        package_json: view(&package_json),
        envelope_json: view(&envelope_json),
        output_ids_json: view(&output_ids_json),
        artifact_handles_json: view(&handles_json),
        run_id: view(b"run:multi-output.predict"),
        controller_bindings: &binding,
        controller_binding_count: 1,
    };
    let mut out = DagMlOwnedBytes::default();
    let mut error = DagMlString::default();
    let status =
        unsafe { dagml_initial_full_refit_predict_json(&predict_request, &mut out, &mut error) };
    assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(error));
    let replay = owned_json(out);
    let outputs = replay["replay_outcome"]["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(
        outputs
            .iter()
            .map(|output| output["output_id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        output_ids.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert_eq!(state.calls, 4);
}

unsafe extern "C" fn release_bytes(_user_data: *mut c_void, bytes: DagMlOwnedBytes) {
    if !bytes.ptr.is_null() {
        drop(Vec::from_raw_parts(bytes.ptr, bytes.len, bytes.capacity));
    }
}

#[test]
fn c_abi_initial_full_refit_then_predict_reuses_signed_package() {
    let package = InitialFullRefitPackage::from_json(PACKAGE).unwrap();
    let plan_json = serde_json::to_vec(&package.effective_plan).unwrap();
    let envelope_json = serde_json::to_vec(&package.training_envelope).unwrap();
    let manifests_json = serde_json::to_vec(
        &package
            .effective_plan
            .controller_manifests
            .values()
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let ids_json = serde_json::to_vec(&package.training_sample_ids).unwrap();
    let mut state = ControllerState::default();
    let binding = DagMlControllerBinding {
        controller_id: view(b"controller:model.mock"),
        vtable: DagMlControllerVTable {
            abi_version: DAG_ML_CONTROLLER_VTABLE_BORROWED_ABI_VERSION,
            user_data: (&mut state as *mut ControllerState).cast(),
            clone_with: None,
            describe: None,
            fit: None,
            predict: None,
            invoke: Some(invoke),
            release_bytes: Some(release_bytes),
            release: None,
            destroy: None,
        },
    };
    let request = DagMlInitialFullRefitExecuteRequest {
        plan_json: view(&plan_json),
        envelope_json: view(&envelope_json),
        trusted_controllers_json: view(&manifests_json),
        training_sample_ids_json: view(&ids_json),
        package_id: view(package.package_id.as_bytes()),
        run_id: view(package.run_id.as_str().as_bytes()),
        root_seed: package.execution_root_seed.unwrap(),
        controller_bindings: &binding,
        controller_binding_count: 1,
    };
    let mut out = DagMlOwnedBytes::default();
    let mut error = DagMlString::default();
    let status = unsafe { dagml_initial_full_refit_execute_json(&request, &mut out, &mut error) };
    assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(error));
    let captured = owned_json(out);
    let captured_package_json =
        serde_json::to_vec(&captured["initial_full_refit_package"]).unwrap();
    let captured_package =
        InitialFullRefitPackage::from_json(std::str::from_utf8(&captured_package_json).unwrap())
            .unwrap();
    assert_eq!(
        captured_package.training_sample_ids,
        package.training_sample_ids
    );
    assert_eq!(state.calls, 1);

    let heldout: SampleRelationSet = serde_json::from_value(json!({"records": [{
        "observation_id": "obs.H001", "sample_id": "sample:heldout:1",
        "target_id": "target:heldout:1", "group_id": "group:heldout",
        "origin_sample_id": null, "source_id": "nir", "is_augmented": false
    }]}))
    .unwrap();
    let cohort = PredictCohort::from_relations(
        PredictCohortRole::ExternalTest,
        heldout,
        vec!["y".into()],
        "a".repeat(64),
        Some("b".repeat(64)),
    )
    .unwrap();
    let predict_envelope = captured_package.predict_envelope(cohort).unwrap();
    let predict_envelope_json = serde_json::to_vec(&predict_envelope).unwrap();
    let output_ids =
        serde_json::to_vec(&vec![captured_package.outputs[0].output_id.clone()]).unwrap();
    let handles = serde_json::to_vec(&captured["node_results"][0]["artifact_handles"]).unwrap();
    let predict_request = DagMlInitialFullRefitPredictRequest {
        package_json: view(&captured_package_json),
        envelope_json: view(&predict_envelope_json),
        output_ids_json: view(&output_ids),
        artifact_handles_json: view(&handles),
        run_id: view(b"run:cabi.initial.predict"),
        controller_bindings: &binding,
        controller_binding_count: 1,
    };
    let mut out = DagMlOwnedBytes::default();
    let mut error = DagMlString::default();
    let status =
        unsafe { dagml_initial_full_refit_predict_json(&predict_request, &mut out, &mut error) };
    assert_eq!(status, DagMlStatusCode::OK, "{}", error_message(error));
    let replay = owned_json(out);
    assert_eq!(
        replay["replay_outcome"]["outputs"][0]["prediction"]["sample_ids"],
        json!(["sample:heldout:1"])
    );
    assert_eq!(state.calls, 2);

    let mut forged: Value = captured["initial_full_refit_package"].clone();
    forged["outputs"][0]["output_id"] = json!("output:forged");
    let forged = serde_json::to_vec(&forged).unwrap();
    let forged_request = DagMlInitialFullRefitPredictRequest {
        package_json: view(&forged),
        ..predict_request
    };
    let mut out = DagMlOwnedBytes::default();
    let mut error = DagMlString::default();
    let status =
        unsafe { dagml_initial_full_refit_predict_json(&forged_request, &mut out, &mut error) };
    assert_eq!(status, DagMlStatusCode::VALIDATION_ERROR);
    assert!(out.ptr.is_null());
    assert!(error_message(error).contains("fingerprint"));
    assert_eq!(
        state.calls, 2,
        "tampered package must fail before controller callback"
    );
}
