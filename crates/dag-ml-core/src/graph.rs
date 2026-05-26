use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::ids::NodeId;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Transform,
    YTransform,
    Split,
    Model,
    Fork,
    Map,
    FeatureJoin,
    PredictionJoin,
    MixedJoin,
    SourceJoin,
    Tag,
    Exclude,
    Augmentation,
    Adapter,
    Aggregator,
    Generator,
    Restructure,
    Tuner,
    Subgraph,
    Chart,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortKind {
    Data,
    Target,
    Prediction,
    Artifact,
    Metric,
    Control,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortCardinality {
    One,
    Many,
    Optional,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortSpec {
    pub name: String,
    pub kind: PortKind,
    pub representation: Option<String>,
    pub cardinality: PortCardinality,
    #[serde(default)]
    pub description: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortSchema {
    #[serde(default)]
    pub inputs: Vec<PortSpec>,
    #[serde(default)]
    pub outputs: Vec<PortSpec>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PortRef {
    pub node_id: NodeId,
    pub port_name: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EdgeContract {
    pub kind: PortKind,
    pub representation: Option<String>,
    #[serde(default)]
    pub requires_oof: bool,
    #[serde(default)]
    pub requires_fold_alignment: bool,
    #[serde(default = "default_true")]
    pub propagates_lineage: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EdgeSpec {
    pub source: PortRef,
    pub target: PortRef,
    pub contract: EdgeContract,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct GraphInterface {
    #[serde(default)]
    pub inputs: Vec<PortSpec>,
    #[serde(default)]
    pub outputs: Vec<PortSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeSpec {
    pub id: NodeId,
    pub kind: NodeKind,
    pub operator: Option<serde_json::Value>,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub ports: PortSchema,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub seed_label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphSpec {
    pub id: String,
    #[serde(default)]
    pub interface: GraphInterface,
    #[serde(default)]
    pub nodes: Vec<NodeSpec>,
    #[serde(default)]
    pub edges: Vec<EdgeSpec>,
    #[serde(default)]
    pub search_space_fingerprint: Option<String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl GraphSpec {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(DagMlError::GraphValidation(
                "graph id must not be empty".to_string(),
            ));
        }
        if self.nodes.is_empty() {
            return Err(DagMlError::GraphValidation(
                "graph must contain at least one node".to_string(),
            ));
        }

        let mut nodes = BTreeMap::new();
        for node in &self.nodes {
            if nodes.insert(node.id.clone(), node).is_some() {
                return Err(DagMlError::GraphValidation(format!(
                    "duplicate node id `{}`",
                    node.id
                )));
            }
            validate_unique_ports(&node.id, "input", &node.ports.inputs)?;
            validate_unique_ports(&node.id, "output", &node.ports.outputs)?;
        }

        let mut adjacency: BTreeMap<NodeId, Vec<NodeId>> = nodes
            .keys()
            .cloned()
            .map(|id| (id, Vec::new()))
            .collect::<BTreeMap<_, _>>();
        let mut indegree: BTreeMap<NodeId, usize> =
            nodes.keys().cloned().map(|id| (id, 0)).collect();

        for edge in &self.edges {
            let source = nodes.get(&edge.source.node_id).ok_or_else(|| {
                DagMlError::GraphValidation(format!(
                    "edge source node `{}` does not exist",
                    edge.source.node_id
                ))
            })?;
            let target = nodes.get(&edge.target.node_id).ok_or_else(|| {
                DagMlError::GraphValidation(format!(
                    "edge target node `{}` does not exist",
                    edge.target.node_id
                ))
            })?;

            let source_port =
                find_port(&source.ports.outputs, &edge.source.port_name).ok_or_else(|| {
                    DagMlError::GraphValidation(format!(
                        "source port `{}.{}` does not exist",
                        edge.source.node_id, edge.source.port_name
                    ))
                })?;
            let target_port =
                find_port(&target.ports.inputs, &edge.target.port_name).ok_or_else(|| {
                    DagMlError::GraphValidation(format!(
                        "target port `{}.{}` does not exist",
                        edge.target.node_id, edge.target.port_name
                    ))
                })?;

            if source_port.kind != edge.contract.kind || target_port.kind != edge.contract.kind {
                return Err(DagMlError::GraphValidation(format!(
                    "edge `{}.{}` -> `{}.{}` has kind {:?}, but ports are {:?} and {:?}",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name,
                    edge.contract.kind,
                    source_port.kind,
                    target_port.kind
                )));
            }
            if edge.contract.requires_oof && edge.contract.kind != PortKind::Prediction {
                return Err(DagMlError::GraphValidation(format!(
                    "edge `{}.{}` -> `{}.{}` requires OOF but is not a prediction edge",
                    edge.source.node_id,
                    edge.source.port_name,
                    edge.target.node_id,
                    edge.target.port_name
                )));
            }

            adjacency
                .get_mut(&edge.source.node_id)
                .expect("source exists")
                .push(edge.target.node_id.clone());
            *indegree
                .get_mut(&edge.target.node_id)
                .expect("target exists") += 1;
        }

        ensure_acyclic(adjacency, indegree)
    }

    pub fn topological_order(&self) -> Result<Vec<NodeId>> {
        self.validate()?;
        let nodes = self
            .nodes
            .iter()
            .map(|node| node.id.clone())
            .collect::<BTreeSet<_>>();
        let mut adjacency = nodes
            .iter()
            .cloned()
            .map(|id| (id, Vec::new()))
            .collect::<BTreeMap<_, _>>();
        let mut indegree: BTreeMap<NodeId, usize> =
            nodes.iter().cloned().map(|id| (id, 0usize)).collect();
        for edge in &self.edges {
            adjacency
                .get_mut(&edge.source.node_id)
                .expect("source exists after validate")
                .push(edge.target.node_id.clone());
            *indegree
                .get_mut(&edge.target.node_id)
                .expect("target exists after validate") += 1;
        }
        topological_order(adjacency, indegree)
    }

    pub fn upstream_nodes(&self, node_id: &NodeId) -> Vec<NodeId> {
        let mut upstream = self
            .edges
            .iter()
            .filter_map(|edge| {
                (edge.target.node_id == *node_id).then_some(edge.source.node_id.clone())
            })
            .collect::<Vec<_>>();
        upstream.sort();
        upstream.dedup();
        upstream
    }

    pub fn downstream_nodes(&self, node_id: &NodeId) -> Vec<NodeId> {
        let mut downstream = self
            .edges
            .iter()
            .filter_map(|edge| {
                (edge.source.node_id == *node_id).then_some(edge.target.node_id.clone())
            })
            .collect::<Vec<_>>();
        downstream.sort();
        downstream.dedup();
        downstream
    }
}

fn validate_unique_ports(node_id: &NodeId, direction: &str, ports: &[PortSpec]) -> Result<()> {
    let mut seen = BTreeSet::new();
    for port in ports {
        if port.name.trim().is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "{} port on node `{}` has an empty name",
                direction, node_id
            )));
        }
        if !seen.insert(port.name.as_str()) {
            return Err(DagMlError::GraphValidation(format!(
                "duplicate {} port `{}` on node `{}`",
                direction, port.name, node_id
            )));
        }
    }
    Ok(())
}

fn find_port<'a>(ports: &'a [PortSpec], name: &str) -> Option<&'a PortSpec> {
    ports.iter().find(|port| port.name == name)
}

fn ensure_acyclic(
    adjacency: BTreeMap<NodeId, Vec<NodeId>>,
    indegree: BTreeMap<NodeId, usize>,
) -> Result<()> {
    topological_order(adjacency, indegree).map(|_| ())
}

fn topological_order(
    adjacency: BTreeMap<NodeId, Vec<NodeId>>,
    mut indegree: BTreeMap<NodeId, usize>,
) -> Result<Vec<NodeId>> {
    let mut queue = indegree
        .iter()
        .filter_map(|(id, degree)| (*degree == 0).then_some(id.clone()))
        .collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(indegree.len());

    while let Some(node) = queue.pop_first() {
        order.push(node.clone());
        if let Some(next_nodes) = adjacency.get(&node) {
            for next in next_nodes {
                let degree = indegree.get_mut(next).expect("node exists");
                *degree -= 1;
                if *degree == 0 {
                    queue.insert(next.clone());
                }
            }
        }
    }

    if order.len() == indegree.len() {
        Ok(order)
    } else {
        Err(DagMlError::GraphValidation(
            "graph contains at least one cycle".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn port(name: &str, kind: PortKind) -> PortSpec {
        PortSpec {
            name: name.to_string(),
            kind,
            representation: None,
            cardinality: PortCardinality::One,
            description: String::new(),
        }
    }

    fn node(id: &str, inputs: Vec<PortSpec>, outputs: Vec<PortSpec>) -> NodeSpec {
        NodeSpec {
            id: NodeId::new(id).unwrap(),
            kind: NodeKind::Model,
            operator: None,
            params: BTreeMap::new(),
            ports: PortSchema { inputs, outputs },
            metadata: BTreeMap::new(),
            seed_label: None,
        }
    }

    fn edge(source: &str, source_port: &str, target: &str, target_port: &str) -> EdgeSpec {
        EdgeSpec {
            source: PortRef {
                node_id: NodeId::new(source).unwrap(),
                port_name: source_port.to_string(),
            },
            target: PortRef {
                node_id: NodeId::new(target).unwrap(),
                port_name: target_port.to_string(),
            },
            contract: EdgeContract {
                kind: PortKind::Prediction,
                representation: None,
                requires_oof: true,
                requires_fold_alignment: true,
                propagates_lineage: true,
            },
        }
    }

    #[test]
    fn validates_simple_graph() {
        let graph = GraphSpec {
            id: "g".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![
                node("model:a", vec![], vec![port("pred", PortKind::Prediction)]),
                node("model:b", vec![port("pred", PortKind::Prediction)], vec![]),
            ],
            edges: vec![edge("model:a", "pred", "model:b", "pred")],
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        };

        assert!(graph.validate().is_ok());
    }

    #[test]
    fn rejects_missing_edge_endpoint() {
        let graph = GraphSpec {
            id: "g".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![node(
                "model:a",
                vec![],
                vec![port("pred", PortKind::Prediction)],
            )],
            edges: vec![edge("model:a", "pred", "model:b", "pred")],
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        };

        assert!(graph.validate().is_err());
    }

    #[test]
    fn rejects_oof_contract_on_non_prediction_edge() {
        let graph = GraphSpec {
            id: "g".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![
                node("model:a", vec![], vec![port("x", PortKind::Data)]),
                node("model:b", vec![port("x", PortKind::Data)], vec![]),
            ],
            edges: vec![EdgeSpec {
                source: PortRef {
                    node_id: NodeId::new("model:a").unwrap(),
                    port_name: "x".to_string(),
                },
                target: PortRef {
                    node_id: NodeId::new("model:b").unwrap(),
                    port_name: "x".to_string(),
                },
                contract: EdgeContract {
                    kind: PortKind::Data,
                    representation: None,
                    requires_oof: true,
                    requires_fold_alignment: true,
                    propagates_lineage: true,
                },
            }],
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        };

        let error = graph.validate().unwrap_err().to_string();

        assert!(error.contains("requires OOF"));
    }

    #[test]
    fn rejects_cycles() {
        let graph = GraphSpec {
            id: "g".to_string(),
            interface: GraphInterface::default(),
            nodes: vec![
                node(
                    "model:a",
                    vec![port("pred", PortKind::Prediction)],
                    vec![port("pred", PortKind::Prediction)],
                ),
                node(
                    "model:b",
                    vec![port("pred", PortKind::Prediction)],
                    vec![port("pred", PortKind::Prediction)],
                ),
            ],
            edges: vec![
                edge("model:a", "pred", "model:b", "pred"),
                edge("model:b", "pred", "model:a", "pred"),
            ],
            search_space_fingerprint: None,
            metadata: BTreeMap::new(),
        };

        assert!(graph.validate().is_err());
    }
}
