use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::bundle::ExecutionBundle;
use crate::data::ExternalDataPlanEnvelope;
use crate::error::{DagMlError, Result};
use crate::ids::{ArtifactId, LineageId};
use crate::plan::ExecutionPlan;
use crate::runtime::{
    FileArtifactManifest, FilePredictionCacheManifest, LineageRecord, FILE_ARTIFACT_MANIFEST_FILE,
    FILE_PREDICTION_CACHE_MANIFEST_FILE,
};

pub const RESEARCH_PROVENANCE_SCHEMA_VERSION: u32 = 1;
pub const PROV_JSONLD_FILE: &str = "lineage.prov.jsonld";
pub const RO_CRATE_METADATA_FILE: &str = "ro-crate-metadata.json";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResearchProvenanceExport {
    pub schema_version: u32,
    pub prov_jsonld: Value,
    pub ro_crate_metadata: Value,
}

pub fn build_research_provenance_export(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    lineage: &[LineageRecord],
    data_envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    prediction_cache_manifest: Option<&FilePredictionCacheManifest>,
    artifact_manifest: Option<&FileArtifactManifest>,
) -> Result<ResearchProvenanceExport> {
    validate_provenance_inputs(
        plan,
        bundle,
        lineage,
        data_envelopes,
        prediction_cache_manifest,
        artifact_manifest,
    )?;

    Ok(ResearchProvenanceExport {
        schema_version: RESEARCH_PROVENANCE_SCHEMA_VERSION,
        prov_jsonld: build_prov_jsonld(
            plan,
            bundle,
            lineage,
            data_envelopes,
            prediction_cache_manifest,
            artifact_manifest,
        )?,
        ro_crate_metadata: build_ro_crate_metadata(
            plan,
            bundle,
            data_envelopes,
            prediction_cache_manifest,
            artifact_manifest,
        )?,
    })
}

fn validate_provenance_inputs(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    lineage: &[LineageRecord],
    data_envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    prediction_cache_manifest: Option<&FilePredictionCacheManifest>,
    artifact_manifest: Option<&FileArtifactManifest>,
) -> Result<()> {
    plan.validate()?;
    bundle.validate_against_plan(plan)?;
    if !data_envelopes.is_empty() {
        bundle.validate_replay_envelopes(data_envelopes)?;
    }
    if let Some(manifest) = prediction_cache_manifest {
        manifest.validate_against_bundle(bundle)?;
    }
    if let Some(manifest) = artifact_manifest {
        manifest.validate_against_bundle(bundle)?;
    }

    let mut lineage_ids = BTreeSet::<&LineageId>::new();
    for record in lineage {
        record.validate()?;
        if !plan.node_plans.contains_key(&record.node_id) {
            return Err(DagMlError::RuntimeValidation(format!(
                "provenance lineage `{}` references unknown node `{}`",
                record.record_id, record.node_id
            )));
        }
        if !plan
            .controller_manifests
            .contains_key(&record.controller_id)
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "provenance lineage `{}` references unknown controller `{}`",
                record.record_id, record.controller_id
            )));
        }
        if !lineage_ids.insert(&record.record_id) {
            return Err(DagMlError::RuntimeValidation(format!(
                "duplicate provenance lineage record `{}`",
                record.record_id
            )));
        }
    }
    for record in lineage {
        for input_id in &record.input_lineage {
            if !lineage_ids.contains(input_id) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "provenance lineage `{}` references missing input lineage `{}`",
                    record.record_id, input_id
                )));
            }
        }
    }
    Ok(())
}

fn build_prov_jsonld(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    lineage: &[LineageRecord],
    data_envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    prediction_cache_manifest: Option<&FilePredictionCacheManifest>,
    artifact_manifest: Option<&FileArtifactManifest>,
) -> Result<Value> {
    let plan_entity_id = format!("dagml:execution-plan:{}", plan.id);
    let bundle_entity_id = format!("dagml:execution-bundle:{}", bundle.bundle_id);
    let packaging_activity_id = format!("dagml:activity:package-bundle:{}", bundle.bundle_id);
    let coordinator_agent_id = "dagml:agent:dag-ml".to_string();

    let mut entity = BTreeMap::<String, Value>::new();
    entity.insert(
        plan_entity_id.clone(),
        json!({
            "prov:type": ["prov:Entity", "dagml:ExecutionPlan"],
            "dagml:plan_id": plan.id,
            "dagml:graph_fingerprint": plan.graph_fingerprint,
            "dagml:campaign_fingerprint": plan.campaign_fingerprint,
            "dagml:controller_fingerprint": plan.controller_fingerprint,
            "dagml:variant_count": plan.variants.len(),
            "dagml:has_fold_set": plan.fold_set.is_some(),
        }),
    );
    entity.insert(
        bundle_entity_id.clone(),
        json!({
            "prov:type": ["prov:Entity", "dagml:ExecutionBundle"],
            "dagml:bundle_id": bundle.bundle_id,
            "dagml:schema_version": bundle.schema_version,
            "dagml:plan_id": bundle.plan_id,
            "dagml:selected_variant_id": bundle.selected_variant_id,
            "dagml:graph_fingerprint": bundle.graph_fingerprint,
            "dagml:campaign_fingerprint": bundle.campaign_fingerprint,
            "dagml:controller_fingerprint": bundle.controller_fingerprint,
            "dagml:unsafe_flags": bundle.unsafe_flags,
            "dagml:selection_count": bundle.selections.len(),
        }),
    );

    for requirement in &bundle.data_requirements {
        let key = requirement.key();
        entity.insert(
            data_requirement_entity_id(&key),
            json!({
                "prov:type": ["prov:Entity", "dagml:DataRequirement"],
                "dagml:requirement_key": key,
                "dagml:node_id": requirement.node_id,
                "dagml:input_name": requirement.input_name,
                "dagml:schema_fingerprint": requirement.schema_fingerprint,
                "dagml:plan_fingerprint": requirement.plan_fingerprint,
                "dagml:relation_fingerprint": requirement.relation_fingerprint,
                "dagml:feature_set_id": requirement.feature_set_id,
            }),
        );
    }
    for (key, envelope) in data_envelopes {
        entity.insert(
            data_envelope_entity_id(key),
            json!({
                "prov:type": ["prov:Entity", "dagml:ExternalDataPlanEnvelope"],
                "dagml:envelope_key": key,
                "dagml:schema_version": envelope.schema_version,
                "dagml:schema_fingerprint": envelope.schema_fingerprint,
                "dagml:plan_fingerprint": envelope.plan_fingerprint,
                "dagml:relation_fingerprint": envelope.relation_fingerprint,
            }),
        );
    }
    for requirement in &bundle.prediction_requirements {
        let key = requirement.key();
        entity.insert(
            prediction_requirement_entity_id(&key),
            json!({
                "prov:type": ["prov:Entity", "dagml:PredictionRequirement"],
                "dagml:requirement_key": key,
                "dagml:producer_node": requirement.producer_node,
                "dagml:consumer_node": requirement.consumer_node,
                "dagml:prediction_level": requirement.prediction_level,
                "dagml:fold_ids": requirement.fold_ids,
                "dagml:unit_ids": requirement.unit_ids,
                "dagml:sample_ids": requirement.sample_ids,
                "dagml:prediction_width": requirement.prediction_width,
                "dagml:target_names": requirement.target_names,
            }),
        );
    }
    for cache in &bundle.prediction_caches {
        entity.insert(
            prediction_cache_entity_id(&cache.cache_id),
            json!({
                "prov:type": ["prov:Entity", "dagml:PredictionCache"],
                "dagml:requirement_key": cache.requirement_key,
                "dagml:cache_id": cache.cache_id,
                "dagml:format": cache.format,
                "dagml:prediction_level": cache.prediction_level,
                "dagml:unit_ids": cache.unit_ids,
                "dagml:block_count": cache.block_count,
                "dagml:row_count": cache.row_count,
                "dagml:content_fingerprint": cache.content_fingerprint,
            }),
        );
    }
    if let Some(manifest) = prediction_cache_manifest {
        entity.insert(
            "dagml:file:prediction-cache-manifest".to_string(),
            json!({
                "prov:type": ["prov:Entity", "dagml:PredictionCacheManifest"],
                "dagml:file": FILE_PREDICTION_CACHE_MANIFEST_FILE,
                "dagml:schema_version": manifest.schema_version,
                "dagml:cache_count": manifest.caches.len(),
            }),
        );
    }
    for record in &bundle.refit_artifacts {
        entity.insert(
            artifact_entity_id(&record.artifact.id),
            json!({
                "prov:type": ["prov:Entity", "dagml:ModelArtifact"],
                "dagml:artifact_id": record.artifact.id,
                "dagml:kind": record.artifact.kind,
                "dagml:node_id": record.node_id,
                "dagml:controller_id": record.controller_id,
                "dagml:backend": record.artifact.backend,
                "dagml:uri": record.artifact.uri,
                "dagml:content_fingerprint": record.artifact.content_fingerprint,
                "dagml:size_bytes": record.artifact.size_bytes,
                "dagml:plugin": record.artifact.plugin,
                "dagml:plugin_version": record.artifact.plugin_version,
                "dagml:params_fingerprint": record.params_fingerprint,
                "dagml:data_requirement_keys": record.data_requirement_keys,
                "dagml:prediction_requirement_keys": record.prediction_requirement_keys,
            }),
        );
    }
    if let Some(manifest) = artifact_manifest {
        entity.insert(
            "dagml:file:artifact-manifest".to_string(),
            json!({
                "prov:type": ["prov:Entity", "dagml:ArtifactManifest"],
                "dagml:file": FILE_ARTIFACT_MANIFEST_FILE,
                "dagml:schema_version": manifest.schema_version,
                "dagml:artifact_count": manifest.artifacts.len(),
            }),
        );
    }

    let mut agent = BTreeMap::<String, Value>::new();
    agent.insert(
        coordinator_agent_id.clone(),
        json!({
            "prov:type": ["prov:Agent", "dagml:Coordinator"],
            "dagml:name": "dag-ml",
            "dagml:provenance_schema_version": RESEARCH_PROVENANCE_SCHEMA_VERSION,
        }),
    );
    for manifest in plan.controller_manifests.values() {
        agent.insert(
            controller_agent_id(manifest.controller_id.as_str()),
            json!({
                "prov:type": ["prov:Agent", "dagml:Controller"],
                "dagml:controller_id": manifest.controller_id,
                "dagml:controller_version": manifest.controller_version,
                "dagml:operator_kind": manifest.operator_kind,
                "dagml:fit_scope": manifest.fit_scope,
                "dagml:rng_policy": manifest.rng_policy,
                "dagml:artifact_policy": manifest.artifact_policy,
                "dagml:capabilities": manifest.capabilities,
            }),
        );
    }

    let mut activity = BTreeMap::<String, Value>::new();
    activity.insert(
        packaging_activity_id.clone(),
        json!({
            "prov:type": ["prov:Activity", "dagml:BundlePackaging"],
            "dagml:bundle_id": bundle.bundle_id,
            "dagml:plan_id": bundle.plan_id,
            "dagml:selected_variant_id": bundle.selected_variant_id,
        }),
    );
    for record in lineage {
        activity.insert(
            lineage_activity_id(record),
            json!({
                "prov:type": ["prov:Activity", "dagml:NodeExecution"],
                "dagml:lineage_id": record.record_id,
                "dagml:run_id": record.run_id,
                "dagml:node_id": record.node_id,
                "dagml:phase": record.phase,
                "dagml:controller_id": record.controller_id,
                "dagml:controller_version": record.controller_version,
                "dagml:variant_id": record.variant_id,
                "dagml:fold_id": record.fold_id,
                "dagml:branch_path": record.branch_path,
                "dagml:params_fingerprint": record.params_fingerprint,
                "dagml:data_model_shape_fingerprint": record.data_model_shape_fingerprint,
                "dagml:aggregation_policy_fingerprint": record.aggregation_policy_fingerprint,
                "dagml:seed": record.seed,
                "dagml:unsafe_flags": record.unsafe_flags,
                "dagml:metrics": record.metrics,
            }),
        );
    }

    let mut used = BTreeMap::<String, Value>::new();
    used.insert(
        "dagml:used:bundle-plan".to_string(),
        json!({
            "prov:activity": packaging_activity_id,
            "prov:entity": plan_entity_id,
        }),
    );
    for record in lineage {
        for input_id in &record.input_lineage {
            used.insert(
                format!("dagml:used:{}:{}", record.record_id, input_id),
                json!({
                    "prov:activity": lineage_activity_id(record),
                    "prov:entity": format!("dagml:activity:{input_id}"),
                    "dagml:input_lineage_id": input_id,
                }),
            );
        }
    }

    let lineage_by_artifact = lineage_artifact_index(lineage);
    let mut was_generated_by = BTreeMap::<String, Value>::new();
    was_generated_by.insert(
        "dagml:generated:bundle".to_string(),
        json!({
            "prov:entity": bundle_entity_id,
            "prov:activity": packaging_activity_id,
        }),
    );
    for record in &bundle.refit_artifacts {
        let activity_id = lineage_by_artifact
            .get(&record.artifact.id)
            .cloned()
            .unwrap_or_else(|| packaging_activity_id.clone());
        was_generated_by.insert(
            format!("dagml:generated:artifact:{}", record.artifact.id),
            json!({
                "prov:entity": artifact_entity_id(&record.artifact.id),
                "prov:activity": activity_id,
            }),
        );
    }

    let mut was_derived_from = BTreeMap::<String, Value>::new();
    was_derived_from.insert(
        "dagml:derived:bundle-plan".to_string(),
        json!({
            "prov:generatedEntity": bundle_entity_id,
            "prov:usedEntity": plan_entity_id,
        }),
    );
    for record in &bundle.refit_artifacts {
        for key in &record.data_requirement_keys {
            was_derived_from.insert(
                format!("dagml:derived:{}:data:{key}", record.artifact.id),
                json!({
                    "prov:generatedEntity": artifact_entity_id(&record.artifact.id),
                    "prov:usedEntity": data_requirement_entity_id(key),
                    "dagml:refit_dependency": "data_requirement",
                }),
            );
        }
        for key in &record.prediction_requirement_keys {
            was_derived_from.insert(
                format!("dagml:derived:{}:prediction:{key}", record.artifact.id),
                json!({
                    "prov:generatedEntity": artifact_entity_id(&record.artifact.id),
                    "prov:usedEntity": prediction_requirement_entity_id(key),
                    "dagml:refit_dependency": "prediction_requirement",
                    "dagml:oof_dependency": true,
                }),
            );
        }
    }
    for cache in &bundle.prediction_caches {
        was_derived_from.insert(
            format!("dagml:derived:cache:{}", cache.cache_id),
            json!({
                "prov:generatedEntity": prediction_cache_entity_id(&cache.cache_id),
                "prov:usedEntity": prediction_requirement_entity_id(&cache.requirement_key),
            }),
        );
    }

    let mut was_associated_with = BTreeMap::<String, Value>::new();
    was_associated_with.insert(
        "dagml:associated:bundle-packaging".to_string(),
        json!({
            "prov:activity": packaging_activity_id,
            "prov:agent": coordinator_agent_id,
        }),
    );
    for record in lineage {
        was_associated_with.insert(
            format!("dagml:associated:{}", record.record_id),
            json!({
                "prov:activity": lineage_activity_id(record),
                "prov:agent": controller_agent_id(record.controller_id.as_str()),
            }),
        );
    }

    Ok(json!({
        "@context": {
            "prov": "http://www.w3.org/ns/prov#",
            "dagml": "https://dag-ml.dev/ns#",
        },
        "entity": entity,
        "activity": activity,
        "agent": agent,
        "used": used,
        "wasGeneratedBy": was_generated_by,
        "wasDerivedFrom": was_derived_from,
        "wasAssociatedWith": was_associated_with,
    }))
}

fn build_ro_crate_metadata(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    data_envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    prediction_cache_manifest: Option<&FilePredictionCacheManifest>,
    artifact_manifest: Option<&FileArtifactManifest>,
) -> Result<Value> {
    let mut has_part = vec![
        json!({"@id": "execution_plan.json"}),
        json!({"@id": "execution_bundle.json"}),
        json!({"@id": PROV_JSONLD_FILE}),
    ];
    let mut graph = vec![
        json!({
            "@id": RO_CRATE_METADATA_FILE,
            "@type": "CreativeWork",
            "about": {"@id": "./"},
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
        }),
        json!({
            "@id": "./",
            "@type": "Dataset",
            "name": format!("DAG-ML research bundle {}", bundle.bundle_id),
            "mainEntity": {"@id": "#workflow"},
            "hasPart": has_part.clone(),
            "dagml:schema_version": RESEARCH_PROVENANCE_SCHEMA_VERSION,
            "dagml:bundle_id": bundle.bundle_id,
            "dagml:plan_id": plan.id,
            "dagml:unsafe_flags": bundle.unsafe_flags,
        }),
        json!({
            "@id": "#workflow",
            "@type": ["ComputationalWorkflow", "SoftwareSourceCode"],
            "name": "DAG-ML compiled workflow",
            "programmingLanguage": "Rust",
            "dagml:plan_id": plan.id,
            "dagml:graph_fingerprint": plan.graph_fingerprint,
            "dagml:campaign_fingerprint": plan.campaign_fingerprint,
            "dagml:controller_fingerprint": plan.controller_fingerprint,
            "dagml:selected_variant_id": bundle.selected_variant_id,
            "dagml:variant_count": plan.variants.len(),
        }),
        file_entity(
            "execution_plan.json",
            "DAG-ML execution plan",
            "dagml:ExecutionPlan",
        ),
        file_entity(
            "execution_bundle.json",
            "DAG-ML execution bundle",
            "dagml:ExecutionBundle",
        ),
        file_entity(PROV_JSONLD_FILE, "DAG-ML W3C PROV export", "prov:Bundle"),
    ];

    if prediction_cache_manifest.is_some() {
        has_part.push(json!({"@id": FILE_PREDICTION_CACHE_MANIFEST_FILE}));
        graph.push(file_entity(
            FILE_PREDICTION_CACHE_MANIFEST_FILE,
            "DAG-ML prediction cache manifest",
            "dagml:PredictionCacheManifest",
        ));
    }
    if artifact_manifest.is_some() {
        has_part.push(json!({"@id": FILE_ARTIFACT_MANIFEST_FILE}));
        graph.push(file_entity(
            FILE_ARTIFACT_MANIFEST_FILE,
            "DAG-ML artifact manifest",
            "dagml:ArtifactManifest",
        ));
    }
    for (key, envelope) in data_envelopes {
        let id = format!("data_envelopes/{key}.json");
        has_part.push(json!({"@id": id}));
        graph.push(json!({
            "@id": id,
            "@type": ["File", "dagml:ExternalDataPlanEnvelope"],
            "name": format!("DAG-ML data envelope {key}"),
            "dagml:envelope_key": key,
            "dagml:schema_version": envelope.schema_version,
            "dagml:schema_fingerprint": envelope.schema_fingerprint,
            "dagml:plan_fingerprint": envelope.plan_fingerprint,
            "dagml:relation_fingerprint": envelope.relation_fingerprint,
        }));
    }

    graph[1]["hasPart"] = Value::Array(has_part);

    for manifest in plan.controller_manifests.values() {
        graph.push(json!({
            "@id": controller_agent_id(manifest.controller_id.as_str()),
            "@type": ["SoftwareApplication", "dagml:Controller"],
            "name": manifest.controller_id,
            "softwareVersion": manifest.controller_version,
            "dagml:operator_kind": manifest.operator_kind,
            "dagml:capabilities": manifest.capabilities,
            "dagml:artifact_policy": manifest.artifact_policy,
        }));
    }
    for artifact in &bundle.refit_artifacts {
        graph.push(json!({
            "@id": artifact_entity_id(&artifact.artifact.id),
            "@type": ["File", "dagml:ModelArtifact"],
            "name": artifact.artifact.id,
            "encodingFormat": artifact.artifact.kind,
            "dagml:node_id": artifact.node_id,
            "dagml:controller_id": artifact.controller_id,
            "dagml:backend": artifact.artifact.backend,
            "dagml:uri": artifact.artifact.uri,
            "dagml:content_fingerprint": artifact.artifact.content_fingerprint,
            "dagml:plugin": artifact.artifact.plugin,
            "dagml:plugin_version": artifact.artifact.plugin_version,
            "dagml:refit_data_requirement_keys": artifact.data_requirement_keys,
            "dagml:refit_prediction_requirement_keys": artifact.prediction_requirement_keys,
        }));
    }

    Ok(json!({
        "@context": [
            "https://w3id.org/ro/crate/1.1/context",
            {
                "dagml": "https://dag-ml.dev/ns#",
                "prov": "http://www.w3.org/ns/prov#",
            }
        ],
        "@graph": graph,
    }))
}

fn file_entity(id: &str, name: &str, dagml_type: &str) -> Value {
    json!({
        "@id": id,
        "@type": ["File", dagml_type],
        "name": name,
    })
}

fn lineage_artifact_index(lineage: &[LineageRecord]) -> BTreeMap<ArtifactId, String> {
    let mut index = BTreeMap::new();
    for record in lineage {
        for artifact in &record.artifact_refs {
            index.insert(artifact.id.clone(), lineage_activity_id(record));
        }
    }
    index
}

fn lineage_activity_id(record: &LineageRecord) -> String {
    format!("dagml:activity:{}", record.record_id)
}

fn controller_agent_id(controller_id: &str) -> String {
    format!("dagml:controller:{controller_id}")
}

fn artifact_entity_id(artifact_id: &ArtifactId) -> String {
    format!("dagml:artifact:{artifact_id}")
}

fn data_requirement_entity_id(key: &str) -> String {
    format!("dagml:data-requirement:{key}")
}

fn data_envelope_entity_id(key: &str) -> String {
    format!("dagml:data-envelope:{key}")
}

fn prediction_requirement_entity_id(key: &str) -> String {
    format!("dagml:prediction-requirement:{key}")
}

fn prediction_cache_entity_id(cache_id: &str) -> String {
    format!("dagml:prediction-cache:{cache_id}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::{ControllerManifest, ControllerRegistry};
    use crate::ids::{ControllerId, LineageId, NodeId, RunId};
    use crate::plan::build_execution_plan;
    use crate::{CampaignSpec, GraphSpec, Phase};

    fn fixture_plan() -> ExecutionPlan {
        let graph: GraphSpec =
            serde_json::from_str(include_str!("../../../examples/minimal_graph.json")).unwrap();
        let campaign: CampaignSpec = serde_json::from_str(include_str!(
            "../../../examples/campaign_oof_generation.json"
        ))
        .unwrap();
        let manifests: Vec<ControllerManifest> =
            serde_json::from_str(include_str!("../../../examples/controller_manifests.json"))
                .unwrap();
        let mut registry = ControllerRegistry::new();
        for manifest in manifests {
            registry.register(manifest).unwrap();
        }
        build_execution_plan("plan:cli.bundle", graph, campaign, &registry).unwrap()
    }

    fn fixture_bundle() -> ExecutionBundle {
        serde_json::from_str(include_str!(
            "../../../examples/generated/execution_bundle_minimal.json"
        ))
        .unwrap()
    }

    fn fixture_lineage(plan: &ExecutionPlan) -> LineageRecord {
        let node_id = NodeId::new("model:base").unwrap();
        let node_plan = plan.node_plans.get(&node_id).unwrap();
        LineageRecord {
            record_id: LineageId::new("lineage:test:model:base").unwrap(),
            run_id: RunId::new("run:provenance").unwrap(),
            node_id,
            phase: Phase::Refit,
            controller_id: node_plan.controller_id.clone(),
            controller_version: node_plan.controller_version.clone(),
            variant_id: plan
                .variants
                .first()
                .map(|variant| variant.variant_id.clone()),
            fold_id: None,
            branch_path: Vec::new(),
            input_lineage: Vec::new(),
            artifact_refs: vec![fixture_bundle().refit_artifacts[0].artifact.clone()],
            params_fingerprint: node_plan.params_fingerprint.clone(),
            data_model_shape_fingerprint: None,
            aggregation_policy_fingerprint: None,
            seed: Some(42),
            unsafe_flags: BTreeSet::new(),
            metrics: BTreeMap::new(),
        }
    }

    #[test]
    fn research_provenance_export_contains_prov_and_ro_crate_contracts() {
        let plan = fixture_plan();
        let bundle = fixture_bundle();
        let lineage = vec![fixture_lineage(&plan)];
        let export = build_research_provenance_export(
            &plan,
            &bundle,
            &lineage,
            &BTreeMap::new(),
            None,
            None,
        )
        .unwrap();

        assert_eq!(export.schema_version, RESEARCH_PROVENANCE_SCHEMA_VERSION);
        assert!(export.prov_jsonld["@context"]["prov"]
            .as_str()
            .unwrap()
            .contains("prov"));
        assert!(export.prov_jsonld["activity"]
            .as_object()
            .unwrap()
            .contains_key("dagml:activity:lineage:test:model:base"));
        assert!(export.prov_jsonld["agent"]
            .as_object()
            .unwrap()
            .contains_key("dagml:controller:controller:model.mock"));
        assert!(export.prov_jsonld["entity"]
            .as_object()
            .unwrap()
            .contains_key("dagml:artifact:artifact:model:base:refit"));

        let graph = export.ro_crate_metadata["@graph"].as_array().unwrap();
        assert!(graph
            .iter()
            .any(|entry| entry["@type"].to_string().contains("ComputationalWorkflow")));
        assert!(graph
            .iter()
            .any(|entry| entry["@id"] == json!("lineage.prov.jsonld")));
        assert!(graph
            .iter()
            .any(|entry| entry["@id"] == json!("execution_bundle.json")));
    }

    #[test]
    fn research_provenance_export_refuses_unknown_lineage_node() {
        let plan = fixture_plan();
        let bundle = fixture_bundle();
        let mut lineage = fixture_lineage(&plan);
        lineage.node_id = NodeId::new("model:missing").unwrap();

        let error = build_research_provenance_export(
            &plan,
            &bundle,
            &[lineage],
            &BTreeMap::new(),
            None,
            None,
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("unknown node"), "unexpected error: {error}");
    }

    #[test]
    fn research_provenance_export_refuses_mismatched_artifact_manifest() {
        let plan = fixture_plan();
        let bundle = fixture_bundle();
        let mut manifest = FileArtifactManifest {
            bundle_id: bundle.bundle_id.clone(),
            schema_version: crate::runtime::FILE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            artifacts: Vec::new(),
        };
        manifest.bundle_id = crate::ids::BundleId::new("bundle:wrong").unwrap();

        let error = build_research_provenance_export(
            &plan,
            &bundle,
            &[],
            &BTreeMap::new(),
            None,
            Some(&manifest),
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("does not match bundle"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn research_provenance_export_refuses_unknown_lineage_controller() {
        let plan = fixture_plan();
        let bundle = fixture_bundle();
        let mut lineage = fixture_lineage(&plan);
        lineage.controller_id = ControllerId::new("controller:missing").unwrap();

        let error = build_research_provenance_export(
            &plan,
            &bundle,
            &[lineage],
            &BTreeMap::new(),
            None,
            None,
        )
        .unwrap_err()
        .to_string();

        assert!(
            error.contains("unknown controller"),
            "unexpected error: {error}"
        );
    }
}
