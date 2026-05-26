use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::graph::{NodeKind, NodeSpec, PortKind, PortSpec};
use crate::ids::ControllerId;
use crate::phase::Phase;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerCapability {
    Deterministic,
    ThreadSafe,
    ProcessSafe,
    NeedsPythonGil,
    EmitsPredictions,
    ConsumesOofPredictions,
    EmitsArtifacts,
    Stateful,
    EmitsRelation,
    UsesCoreRng,
    ShapeChanging,
    GeneratesData,
    GeneratesModel,
    ExpandsVariants,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ControllerFitScope {
    Stateless,
    FoldTrain,
    FullTrain,
    InferenceOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RngPolicy {
    UsesCoreSeed,
    IgnoresSeed,
    ExternallyDeterministic,
    Nondeterministic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactPolicy {
    Serializable,
    HostOnly,
    ContentAddressed,
    ReplayRequired,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ControllerManifest {
    pub controller_id: ControllerId,
    pub controller_version: String,
    pub operator_kind: NodeKind,
    #[serde(default)]
    pub priority: u32,
    #[serde(default)]
    pub supported_phases: BTreeSet<Phase>,
    #[serde(default)]
    pub input_ports: Vec<PortSpec>,
    #[serde(default)]
    pub output_ports: Vec<PortSpec>,
    #[serde(default)]
    pub data_requirements: Option<serde_json::Value>,
    #[serde(default)]
    pub capabilities: BTreeSet<ControllerCapability>,
    pub fit_scope: ControllerFitScope,
    pub rng_policy: RngPolicy,
    pub artifact_policy: ArtifactPolicy,
}

impl ControllerManifest {
    pub fn validate(&self) -> Result<()> {
        if self.controller_version.trim().is_empty() {
            return Err(DagMlError::ControllerValidation(format!(
                "controller `{}` has an empty version",
                self.controller_id
            )));
        }
        if self.supported_phases.is_empty() {
            return Err(DagMlError::ControllerValidation(format!(
                "controller `{}` supports no phases",
                self.controller_id
            )));
        }
        validate_ports(&self.controller_id, "input", &self.input_ports)?;
        validate_ports(&self.controller_id, "output", &self.output_ports)?;
        if self.rng_policy == RngPolicy::Nondeterministic
            && self
                .capabilities
                .contains(&ControllerCapability::Deterministic)
        {
            return Err(DagMlError::ControllerValidation(format!(
                "controller `{}` cannot be deterministic with nondeterministic RNG",
                self.controller_id
            )));
        }
        if self.fit_scope == ControllerFitScope::InferenceOnly
            && (self.supported_phases.contains(&Phase::FitCv)
                || self.supported_phases.contains(&Phase::Refit))
        {
            return Err(DagMlError::ControllerValidation(format!(
                "controller `{}` is inference_only but supports training phases",
                self.controller_id
            )));
        }
        if self.supported_phases.contains(&Phase::FitCv)
            && matches!(
                self.fit_scope,
                ControllerFitScope::FullTrain | ControllerFitScope::InferenceOnly
            )
        {
            return Err(DagMlError::ControllerValidation(format!(
                "controller `{}` supports FIT_CV but has fit_scope {:?}",
                self.controller_id, self.fit_scope
            )));
        }
        if self
            .output_ports
            .iter()
            .any(|port| port.kind == PortKind::Prediction)
            && !self
                .capabilities
                .contains(&ControllerCapability::EmitsPredictions)
        {
            return Err(DagMlError::ControllerValidation(format!(
                "controller `{}` has prediction output ports but lacks emits_predictions",
                self.controller_id
            )));
        }
        if self
            .output_ports
            .iter()
            .any(|port| port.kind == PortKind::Artifact)
            && !self
                .capabilities
                .contains(&ControllerCapability::EmitsArtifacts)
        {
            return Err(DagMlError::ControllerValidation(format!(
                "controller `{}` has artifact output ports but lacks emits_artifacts",
                self.controller_id
            )));
        }
        Ok(())
    }

    pub fn supports_phase(&self, phase: Phase) -> bool {
        self.supported_phases.contains(&phase)
    }

    pub fn supports_parallel_invocation(&self) -> bool {
        self.capabilities
            .contains(&ControllerCapability::ThreadSafe)
            || self
                .capabilities
                .contains(&ControllerCapability::ProcessSafe)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ControllerRegistry {
    manifests: BTreeMap<ControllerId, ControllerManifest>,
}

impl ControllerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, manifest: ControllerManifest) -> Result<()> {
        manifest.validate()?;
        if self.manifests.contains_key(&manifest.controller_id) {
            return Err(DagMlError::ControllerValidation(format!(
                "duplicate controller id `{}`",
                manifest.controller_id
            )));
        }
        self.manifests
            .insert(manifest.controller_id.clone(), manifest);
        Ok(())
    }

    pub fn get(&self, controller_id: &ControllerId) -> Option<&ControllerManifest> {
        self.manifests.get(controller_id)
    }

    pub fn manifests(&self) -> impl Iterator<Item = &ControllerManifest> {
        self.manifests.values()
    }

    pub fn resolve_for_node(&self, node: &NodeSpec) -> Result<ControllerManifest> {
        if let Some(requested) = requested_controller(node)? {
            let manifest = self.get(&requested).ok_or_else(|| {
                DagMlError::Planning(format!(
                    "node `{}` requested unknown controller `{requested}`",
                    node.id
                ))
            })?;
            if manifest.operator_kind != node.kind {
                return Err(DagMlError::Planning(format!(
                    "node `{}` kind {:?} is incompatible with controller `{}` kind {:?}",
                    node.id, node.kind, manifest.controller_id, manifest.operator_kind
                )));
            }
            return Ok(manifest.clone());
        }

        let mut candidates = self
            .manifests
            .values()
            .filter(|manifest| manifest.operator_kind == node.kind)
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            left.priority
                .cmp(&right.priority)
                .then_with(|| left.controller_id.cmp(&right.controller_id))
        });
        let Some(first) = candidates.first() else {
            return Err(DagMlError::Planning(format!(
                "no controller registered for node `{}` kind {:?}",
                node.id, node.kind
            )));
        };
        if candidates
            .get(1)
            .is_some_and(|second| second.priority == first.priority)
        {
            return Err(DagMlError::Planning(format!(
                "node `{}` has ambiguous controllers for kind {:?}; set metadata.controller_id",
                node.id, node.kind
            )));
        }
        Ok((*first).clone())
    }
}

fn validate_ports(controller_id: &ControllerId, direction: &str, ports: &[PortSpec]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for port in ports {
        if port.name.trim().is_empty() {
            return Err(DagMlError::ControllerValidation(format!(
                "{direction} port on controller `{controller_id}` has an empty name"
            )));
        }
        if !seen.insert(port.name.as_str()) {
            return Err(DagMlError::ControllerValidation(format!(
                "duplicate {direction} port `{}` on controller `{controller_id}`",
                port.name
            )));
        }
    }
    Ok(())
}

fn requested_controller(node: &NodeSpec) -> Result<Option<ControllerId>> {
    node.metadata
        .get("controller_id")
        .map(|value| {
            value.as_str().ok_or_else(|| {
                DagMlError::Planning(format!(
                    "node `{}` metadata.controller_id must be a string",
                    node.id
                ))
            })
        })
        .transpose()?
        .map(ControllerId::new)
        .transpose()
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use serde_json::json;

    use super::*;
    use crate::graph::{NodeSpec, PortCardinality, PortSchema};
    use crate::ids::NodeId;

    fn manifest(id: &str, kind: NodeKind, priority: u32) -> ControllerManifest {
        ControllerManifest {
            controller_id: ControllerId::new(id).unwrap(),
            controller_version: "0.1.0".to_string(),
            operator_kind: kind,
            priority,
            supported_phases: BTreeSet::from([Phase::FitCv]),
            input_ports: Vec::new(),
            output_ports: Vec::new(),
            data_requirements: None,
            capabilities: BTreeSet::from([ControllerCapability::Deterministic]),
            fit_scope: ControllerFitScope::FoldTrain,
            rng_policy: RngPolicy::UsesCoreSeed,
            artifact_policy: ArtifactPolicy::Serializable,
        }
    }

    fn node(kind: NodeKind) -> NodeSpec {
        NodeSpec {
            id: NodeId::new("node:model").unwrap(),
            kind,
            operator: None,
            params: BTreeMap::new(),
            ports: PortSchema::default(),
            metadata: BTreeMap::new(),
            seed_label: None,
        }
    }

    #[test]
    fn registry_resolves_lowest_priority_manifest() {
        let mut registry = ControllerRegistry::new();
        registry
            .register(manifest("controller:slow", NodeKind::Model, 10))
            .unwrap();
        registry
            .register(manifest("controller:fast", NodeKind::Model, 1))
            .unwrap();

        let resolved = registry.resolve_for_node(&node(NodeKind::Model)).unwrap();

        assert_eq!(resolved.controller_id.as_str(), "controller:fast");
    }

    #[test]
    fn explicit_controller_id_disambiguates() {
        let mut registry = ControllerRegistry::new();
        registry
            .register(manifest("controller:a", NodeKind::Model, 1))
            .unwrap();
        registry
            .register(manifest("controller:b", NodeKind::Model, 1))
            .unwrap();
        let mut node = node(NodeKind::Model);
        node.metadata
            .insert("controller_id".to_string(), json!("controller:b"));

        let resolved = registry.resolve_for_node(&node).unwrap();

        assert_eq!(resolved.controller_id.as_str(), "controller:b");
    }

    #[test]
    fn equal_priority_requires_explicit_controller() {
        let mut registry = ControllerRegistry::new();
        registry
            .register(manifest("controller:a", NodeKind::Model, 1))
            .unwrap();
        registry
            .register(manifest("controller:b", NodeKind::Model, 1))
            .unwrap();

        assert!(registry.resolve_for_node(&node(NodeKind::Model)).is_err());
    }

    #[test]
    fn manifest_rejects_prediction_output_without_capability() {
        let mut manifest = manifest("controller:predictor", NodeKind::Model, 0);
        manifest.output_ports.push(PortSpec {
            name: "pred".to_string(),
            kind: PortKind::Prediction,
            representation: None,
            cardinality: PortCardinality::One,
            description: String::new(),
        });

        let error = manifest.validate().unwrap_err().to_string();

        assert!(error.contains("lacks emits_predictions"));
    }

    #[test]
    fn manifest_rejects_training_phases_for_inference_only_controller() {
        let mut manifest = manifest("controller:predict-only", NodeKind::Model, 0);
        manifest.fit_scope = ControllerFitScope::InferenceOnly;

        let error = manifest.validate().unwrap_err().to_string();

        assert!(error.contains("inference_only"));
    }

    #[test]
    fn manifest_reports_parallel_invocation_support() {
        let mut manifest = manifest("controller:parallel", NodeKind::Model, 0);
        assert!(!manifest.supports_parallel_invocation());
        manifest
            .capabilities
            .insert(ControllerCapability::ProcessSafe);
        assert!(manifest.supports_parallel_invocation());
    }
}
