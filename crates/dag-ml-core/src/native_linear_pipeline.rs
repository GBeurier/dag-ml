//! Portable request construction for the bounded single-source native linear pipeline.
//!
//! Moved from Core's `pipeline_workflow.rs` and the Raw, origin-enabled branch
//! of `io_training::base_regression_training_request`. Keep the historical IDs,
//! policies, ordering and serialization: they are part of the request fingerprint.
//! IO/Core extract facts; hosts supply catalog-derived controller manifests.
//! This module neither accesses buffers nor configures a Methods runtime.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    ControllerManifest, DataBinding, ExternalDataPlanEnvelope, GraphSpec, NodeId, Phase,
    SampleRelationSet, TrainingDataIdentity, TrainingRequest, TRAINING_REQUEST_SCHEMA_VERSION,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePipelineStep {
    pub method_id: String,
    pub role: String,
    #[serde(default)]
    pub params: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativePipelineRecipe {
    pub steps: Vec<NativePipelineStep>,
    pub candidates: Vec<BTreeMap<String, Value>>,
}

/// IO-owned identities and declared folds for one selected training source.
///
/// The caller must supply the exact converted provider envelope and relations,
/// including any target-mask content identity. The caller retains IO task/role,
/// source and target-mask validation. Relation order is provider order;
/// target order is IO order. No feature or target buffers are carried here.
#[derive(Clone, Debug)]
pub struct NativeLinearPipelineFacts {
    pub envelope: ExternalDataPlanEnvelope,
    pub relations: SampleRelationSet,
    pub feature_set_id: String,
    pub source_id: String,
    pub target_names: Vec<String>,
    /// The first `input_sample_ids` audit, or `None` when absent. Declared fold
    /// indices address this order; absence uses relation order, as in Core.
    pub input_sample_ids: Option<Vec<String>>,
    /// IO-declared (train, validation) indices; retain their exact order.
    pub declared_folds: Vec<(Vec<i64>, Vec<i64>)>,
}

/// Construct the existing fingerprint-bound linear training request.
///
/// `manifests` are the full catalog-derived manifests supplied by the host;
/// they are sorted by controller ID exactly as in the native Core caller.
/// Role-dependent classification/target-index policy, leakage, selection and
/// portable-refit checks are delegated to the unchanged request path below.
pub fn native_pipeline_training_request(
    facts: &NativeLinearPipelineFacts,
    recipe: &NativePipelineRecipe,
    manifests: &[ControllerManifest],
) -> Result<TrainingRequest, String> {
    if recipe.steps.is_empty()
        || recipe.steps.len() > 32
        || recipe.candidates.is_empty()
        || recipe.candidates.len() > 32
    {
        return Err(
            "native pipeline requires 1..32 steps and 1..32 candidate parameter maps".into(),
        );
    }
    let classification = recipe
        .steps
        .last()
        .is_some_and(|step| step.role == "classifier");
    let base = base_regression_training_request(facts)?;
    let mut manifests = manifests.to_vec();
    manifests.sort_by(|a, b| a.controller_id.cmp(&b.controller_id));
    let mut request = serde_json::to_value(base).map_err(|e| e.to_string())?;
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut ids = Vec::new();
    for (index, step) in recipe.steps.iter().enumerate() {
        let final_step = index + 1 == recipe.steps.len();
        let kind = match (step.role.as_str(), final_step) {
            ("transformer" | "selector", false) => "transform",
            ("regressor" | "classifier", true) => "model",
            _ => return Err(
                "pipeline requires transformer/selector steps followed by one regressor/classifier"
                    .into(),
            ),
        };
        if step.method_id.trim().is_empty()
            || step.params.contains_key("method_id")
            || step.params.contains_key("unsafe_flags")
        {
            return Err(
                "method_id and unsafe_flags cannot be overridden by user parameters".into(),
            );
        }
        let id = if final_step {
            "model:pls".to_string()
        } else {
            format!("transform:{index}")
        };
        NodeId::new(&id).map_err(|e| e.to_string())?;
        let mut params = step.params.clone();
        params.insert("method_id".into(), json!(step.method_id));
        nodes.push(json!({"id":id,"kind":kind,"operator":format!("n4m:{}",step.method_id),"params":params,
            "ports":{"inputs":[{"name":"x","kind":"data","representation":"tabular_numeric","cardinality":"one","description":""}],
            "outputs":[{"name":if final_step {"oof"} else {"x_out"},"kind":if final_step {"prediction"} else {"data"},"representation":if final_step {Value::Null} else {json!("tabular_numeric")},"cardinality":"one","description":""}]},"metadata":{},"seed_label":null}));
        if let Some(previous) = ids.last() {
            edges.push(json!({"source":{"node_id":previous,"port_name":"x_out"},"target":{"node_id":id,"port_name":"x"},
                "contract":{"kind":"data","representation":"tabular_numeric","requires_oof":false,"requires_fold_alignment":true,"propagates_lineage":true}}));
        }
        ids.push(id);
    }
    request["campaign"]["metadata"]["native_pipeline_target_names"] =
        request["options"]["outputs"][0]["target_names"].clone();
    if let Some(index) = recipe
        .steps
        .last()
        .and_then(|step| step.params.get("native_target_index"))
    {
        let index = index
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("native_target_index requires addressable nonnegative integer")?;
        for key in ["target_names", "target_units", "class_labels"] {
            let values = request["options"]["outputs"][0][key]
                .as_array()
                .ok_or("Invalid native output target schema")?;
            let value = values
                .get(index)
                .ok_or("native_target_index exceeds declared targets")?
                .clone();
            request["options"]["outputs"][0][key] = json!([value]);
        }
    }
    if classification {
        if request["options"]["outputs"][0]["target_names"]
            .as_array()
            .is_none_or(|names| names.len() != 1)
        {
            return Err("native classification requires exactly one target".into());
        }
        request["options"]["outputs"][0]["prediction_kind"] = json!("class_label");
        request["options"]["selection"]["metric"] =
            json!({"name":"balanced_accuracy","objective":"maximize"});
    }
    request["graph"]["nodes"] = json!(nodes);
    request["graph"]["edges"] = json!(edges);
    request["controller_manifests"] = json!(manifests);
    // Only the first node reads raw IO buffers; subsequent nodes consume their
    // upstream native identity-keyed feature handles.
    let mut binding = request["campaign"]["data_bindings"]["model:pls"][0].clone();
    binding["node_id"] = json!(ids[0]);
    request["campaign"]["data_bindings"] = json!({ids[0].clone():[binding]});
    let shape = request["campaign"]["shape_plans"]["model:pls"].clone();
    let mut shapes = serde_json::Map::new();
    for id in &ids {
        let mut declaration = shape.clone();
        declaration["node_id"] = json!(id);
        shapes.insert(id.clone(), declaration);
    }
    request["campaign"]["shape_plans"] = Value::Object(shapes);
    let typed_binding: DataBinding =
        serde_json::from_value(request["campaign"]["data_bindings"][&ids[0]][0].clone())
            .map_err(|e| e.to_string())?;
    request["data_identities"] =
        json!([
            TrainingDataIdentity::from_binding_envelope(&typed_binding, &facts.envelope)
                .map_err(|e| e.to_string())?
        ]);
    let sample_ids: Vec<String> = facts.input_sample_ids.clone().unwrap_or_else(|| {
        facts
            .relations
            .records
            .iter()
            .map(|r| r.sample_id.to_string())
            .collect()
    });
    let groups: BTreeMap<String, String> = facts
        .relations
        .records
        .iter()
        .map(|r| {
            (
                r.sample_id.to_string(),
                r.group_id
                    .as_ref()
                    .map(|g| g.to_string())
                    .unwrap_or_else(|| r.sample_id.to_string()),
            )
        })
        .collect();
    let dependent = groups
        .values()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != groups.len()
        || facts
            .relations
            .records
            .iter()
            .any(|r| r.origin_sample_id.is_some());
    if dependent && facts.declared_folds.is_empty() {
        return Err("dependent/grouped samples require explicit leakage-safe fold_ids".into());
    }
    if !facts.declared_folds.is_empty() {
        if facts.declared_folds.len() < 2 {
            return Err("at least two declared CV folds required".into());
        }
        let fold_set = json!({"id":"native:declared-folds","sample_ids":sample_ids,
            "sample_groups":if dependent {json!(groups)} else {json!({})},
            "folds":facts.declared_folds.iter().enumerate().map(|(i,(train,validation))|{
                let select = |indexes:&Vec<i64>| -> Result<Vec<String>,String> {
                    indexes.iter().map(|index| usize::try_from(*index).ok().and_then(|position|sample_ids.get(position)).cloned().ok_or("declared fold index outside native training cohort".into())).collect()
                };
                Ok(json!({"fold_id":format!("native.fold.{i}"),"train_sample_ids":select(train)?,"validation_sample_ids":select(validation)?,"metadata":{}}))
            }).collect::<Result<Vec<_>,String>>()?});
        request["campaign"]["split_invocation"]["fold_set"] = fold_set;
        request["campaign"]["split_invocation"]["id"] = json!("native:declared-folds");
    }
    if dependent {
        request["campaign"]["leakage_policy"]["split_unit"] = json!("group");
        request["campaign"]["leakage_policy"]["require_group_ids"] = json!(true);
        request["campaign"]["split_invocation"]["leakage_policy"] =
            request["campaign"]["leakage_policy"].clone();
    }
    request["campaign"]["metadata"]["native_pipeline_recipe"] = json!(recipe);
    request["campaign"]["metadata"]["input_sample_ids"] = json!(facts
        .relations
        .records
        .iter()
        .map(|r| r.sample_id.to_string())
        .collect::<Vec<_>>());
    for candidate in &recipe.candidates {
        if candidate.contains_key("method_id")
            || candidate.contains_key("unsafe_flags")
            || candidate.contains_key("native_target_index")
        {
            return Err("candidate parameters cannot override method identity or safety".into());
        }
    }
    request["campaign"]["generation"] = json!({"strategy":"cartesian","max_variants":recipe.candidates.len(),
        "dimensions":[{"name":"native_model_params","choices":recipe.candidates.iter().enumerate().map(|(i,params)|json!({
            "label":format!("candidate_{i}"),"value":i,"param_overrides":if params.is_empty() {vec![]} else {vec![json!({"node_id":"model:pls","params":params})]},"active_subsequence":null})).collect::<Vec<_>>()}]});
    let mut request: TrainingRequest =
        serde_json::from_value(request).map_err(|e| e.to_string())?;
    request.request_fingerprint = request.compute_fingerprint().map_err(|e| e.to_string())?;
    request.validate().map_err(|e| e.to_string())?;
    let projected = request.project().map_err(|e| e.to_string())?;
    for (id, step) in ids.iter().zip(&recipe.steps) {
        let node_id = NodeId::new(id).map_err(|e| e.to_string())?;
        let expected = format!("controller:n4m.{}", step.role);
        if projected
            .plan
            .node_plans
            .get(&node_id)
            .is_none_or(|node| node.controller_id.as_str() != expected)
        {
            return Err(format!(
                "native catalog role differs from declared role for {id}"
            ));
        }
    }
    if projected
        .plan
        .node_plans
        .values()
        .any(|node| !node.supported_phases.contains(&Phase::Refit))
    {
        return Err("every native pipeline step must support portable full refit".into());
    }
    Ok(request)
}

// Exact Raw/origin-enabled base policy used by the native linear producer.
fn base_regression_training_request(
    facts: &NativeLinearPipelineFacts,
) -> Result<TrainingRequest, String> {
    let mut sample_ids = Vec::new();
    let mut seen = BTreeSet::new();
    for relation in &facts.relations.records {
        let partitions: Vec<_> = relation
            .tags
            .iter()
            .filter(|tag| tag.starts_with("partition:"))
            .collect();
        if partitions.len() != 1 || partitions[0] != "partition:train" {
            return Err("canonical PLS training requires only train partition rows".into());
        }
        if relation.is_augmented || relation.excluded {
            return Err(
                "canonical PLS requires unaugmented, included, origin-free samples".to_string(),
            );
        }
        let sample_id = relation.sample_id.to_string();
        if !seen.insert(sample_id.clone()) {
            return Err(format!(
                "canonical PLS requires one relation per sample; `{sample_id}` is duplicated"
            ));
        }
        sample_ids.push(sample_id);
    }
    if sample_ids.len() < 4 {
        return Err(format!(
            "canonical PLS requires at least four samples, got {}",
            sample_ids.len()
        ));
    }
    let validation_0 = sample_ids.iter().step_by(2).cloned().collect::<Vec<_>>();
    let training_0 = sample_ids
        .iter()
        .skip(1)
        .step_by(2)
        .cloned()
        .collect::<Vec<_>>();
    let envelope = &facts.envelope;
    let relation_fingerprint = envelope
        .relation_fingerprint
        .clone()
        .ok_or("canonical PLS requires a relation fingerprint")?;
    let binding: DataBinding = serde_json::from_value(serde_json::json!({
        "node_id": "model:pls",
        "input_name": "x",
        "request_id": "io:canonical-raw-pls",
        "schema_fingerprint": envelope.schema_fingerprint,
        "plan_fingerprint": envelope.plan_fingerprint,
        "relation_fingerprint": relation_fingerprint,
        "output_representation": "tabular_numeric",
        "feature_set_id": facts.feature_set_id,
        "source_ids": [facts.source_id],
        "require_relations": true,
        "view_policy": {
            "fit_partition": "fold_train",
            "predict_partition": "fold_validation",
            "include_augmented_train": false,
            "include_augmented_validation": false,
            "include_excluded": false,
            "require_sample_ids": true
        },
        "metadata": {}
    }))
    .map_err(|error| error.to_string())?;
    let identity = TrainingDataIdentity::from_binding_envelope(&binding, envelope)
        .map_err(|error| error.to_string())?;
    let node_params = serde_json::json!({"n_components": 1});
    let graph: GraphSpec = serde_json::from_value(serde_json::json!({
        "id": "core-canonical-raw-pls",
        "interface": {
            "inputs": [{"name": "x", "kind": "data", "representation": "tabular_numeric", "cardinality": "one", "description": "selected IO numeric source"}],
            "outputs": [{"name": "prediction", "kind": "prediction", "representation": null, "cardinality": "one", "description": "PLS prediction"}]
        },
        "nodes": [{
            "id": "model:pls",
            "kind": "model",
            "operator": "pls",
            "params": node_params,
            "ports": {
                "inputs": [{"name": "x", "kind": "data", "representation": "tabular_numeric", "cardinality": "one", "description": ""}],
                "outputs": [{"name": "oof", "kind": "prediction", "representation": null, "cardinality": "one", "description": ""}]
            },
            "metadata": {},
            "seed_label": null
        }],
        "edges": [],
        "search_space_fingerprint": null,
        "metadata": {}
    }))
    .map_err(|error| error.to_string())?;
    let campaign = serde_json::from_value(serde_json::json!({
        "id": "campaign:core:canonical-raw-pls",
        "root_seed": 91,
        "leakage_policy": {
            "split_unit": "sample",
            "forbid_origin_cross_fold": true,
            "allow_observation_split_with_shared_target": false,
            "require_group_ids": false,
            "unsafe_flags": []
        },
        "aggregation_policy": {
            "aggregation_level": "sample",
            "method": "mean",
            "weights": "none",
            "emit_parallel_metrics": true,
            "selection_metric_level": "sample",
            "store_raw_predictions": true,
            "store_aggregated_predictions": true
        },
        "split_invocation": {
            "id": "core:canonical:two-fold",
            "controller_id": null,
            "leakage_policy": {
                "split_unit": "sample",
                "forbid_origin_cross_fold": true,
                "allow_observation_split_with_shared_target": false,
                "require_group_ids": false,
                "unsafe_flags": []
            },
            "params": {"kind": "precomputed"},
            "fold_set": {
                "id": "core:canonical:two-fold",
                "sample_ids": sample_ids,
                "folds": [
                    {"fold_id": "core.fold.0", "train_sample_ids": training_0, "validation_sample_ids": validation_0},
                    {"fold_id": "core.fold.1", "train_sample_ids": validation_0, "validation_sample_ids": training_0}
                ],
                "sample_groups": {}
            }
        },
        "generation": {"strategy": "none", "dimensions": [], "max_variants": 1},
        "shape_plans": {
            "model:pls": {
                "node_id": "model:pls",
                "input_granularity": "sample",
                "target_granularity": "sample",
                "fit_rows": "fold_train",
                "predict_rows": "fold_validation",
                "feature_namespace": facts.source_id,
                "feature_schema_fingerprint": null,
                "target_space": "raw",
                "aggregation_policy": {
                    "aggregation_level": "sample", "method": "mean", "weights": "none",
                    "emit_parallel_metrics": true, "selection_metric_level": "sample",
                    "store_raw_predictions": true, "store_aggregated_predictions": true
                },
                "augmentation_policy": {
                    "sample_scope": "train_only", "feature_scope": "train_only",
                    "require_origin_id": true, "inherit_group": true, "inherit_target": true
                },
                "selection_policy": {"scope": "none", "store_masks": true, "allow_schema_mismatch_on_join": false}
            }
        },
        "data_bindings": {"model:pls": [binding]},
        "metadata": {}
    }))
    .map_err(|error| error.to_string())?;
    let controller_manifests = serde_json::from_value(serde_json::json!([{
        "controller_id": "controller:methods.pls",
        "controller_version": "libn4m-2.5",
        "operator_kind": "model",
        "priority": 0,
        "supported_phases": ["FIT_CV", "REFIT", "PREDICT"],
        "input_ports": [{"name": "x", "kind": "data", "representation": "tabular_numeric", "cardinality": "one", "description": ""}],
        "output_ports": [{"name": "oof", "kind": "prediction", "representation": null, "cardinality": "one", "description": ""}],
        "data_requirements": null,
        "capabilities": ["deterministic", "thread_safe", "process_safe", "emits_predictions", "emits_artifacts", "stateful", "supports_portable_full_refit"],
        "fit_scope": "fold_train",
        "rng_policy": "uses_core_seed",
        "artifact_policy": "serializable"
    }]))
    .map_err(|error| error.to_string())?;
    let target_names = facts.target_names.clone();
    let target_units = vec![None::<String>; target_names.len()];
    let class_labels = vec![Vec::<String>::new(); target_names.len()];
    let options = serde_json::from_value(serde_json::json!({
        "refit": true,
        "refit_strategy": "refit_one",
        "seed": 91,
        "selection": {
            "id": "selection:rmse",
            "metric": {"name": "rmse", "objective": "minimize"},
            "required_metric_level": "sample",
            "require_finite": true,
            "evaluation_scope": "oof"
        },
        "selection_output_id": "output:prediction",
        "outputs": [{
            "output_id": "output:prediction", "node_id": "model:pls", "port_name": "oof",
            "prediction_level": "sample", "unit_level": "physical_sample",
            "prediction_kind": "regression_point", "target_names": target_names,
            "target_units": target_units, "class_labels": class_labels,
            "output_order": "target_order", "target_space": "raw"
        }],
        "scheduler": {"kind": "sequential", "backend": null, "workers": 1},
        "resources": {"cpu_threads": 1, "memory_bytes": null, "gpu_devices": [], "wall_time_ms": null},
        "artifacts": {"cv_artifacts": "discard", "prediction_caches": "retain", "fitted_artifacts": "portable_required"}
    }))
    .map_err(|error| error.to_string())?;
    let mut request = TrainingRequest {
        schema_version: TRAINING_REQUEST_SCHEMA_VERSION,
        request_id: "training:core:canonical-raw-pls".into(),
        plan_id: "plan:core:canonical-raw-pls".into(),
        graph,
        campaign,
        controller_manifests,
        data_identities: vec![identity],
        parameter_patches: vec![],
        patch_policies: vec![],
        influence_requirements: vec![],
        training_losses: vec![],
        options,
        request_fingerprint: "0".repeat(64),
    };
    request.request_fingerprint = request
        .compute_fingerprint()
        .map_err(|error| error.to_string())?;
    request.validate().map_err(|error| error.to_string())?;
    Ok(request)
}
