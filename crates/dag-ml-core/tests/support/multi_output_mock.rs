//! Synthetic host for genuine imported public-producer controls. No numerics or IO qualification.
use dag_ml_core::*;
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Default)]
pub struct Counts {
    pub data: AtomicUsize,
    pub controller: AtomicUsize,
    pub hydrate: AtomicUsize,
    pub release: AtomicUsize,
}
impl Counts {
    pub fn zero(&self) {
        assert_eq!(self.data.load(Ordering::SeqCst), 0);
        assert_eq!(self.controller.load(Ordering::SeqCst), 0);
        assert_eq!(self.hydrate.load(Ordering::SeqCst), 0);
    }
}
pub struct Provider {
    pub envelopes: BTreeMap<String, ExternalDataPlanEnvelope>,
    pub counts: Arc<Counts>,
}
impl RuntimeDataProvider for Provider {
    fn materialize(&self, _: &DataMaterializationRequest) -> Result<HandleRef> {
        self.counts.data.fetch_add(1, Ordering::SeqCst);
        Ok(handle(
            "controller:synthetic.provider",
            900,
            HandleKind::Data,
        ))
    }
    fn make_view(&self, _: &DataViewRequest) -> Result<HandleRef> {
        self.counts.data.fetch_add(1, Ordering::SeqCst);
        Ok(handle(
            "controller:synthetic.provider",
            901,
            HandleKind::DataView,
        ))
    }
    fn coordinator_relations(&self, b: &DataBinding) -> Result<Option<SampleRelationSet>> {
        self.counts.data.fetch_add(1, Ordering::SeqCst);
        Ok(
            self.envelopes[&data_binding_requirement_key(&b.node_id, &b.input_name)]
                .coordinator_relations
                .clone(),
        )
    }
    fn predict_cohort(&self, b: &DataBinding, _: Phase) -> Result<Option<PredictCohort>> {
        self.counts.data.fetch_add(1, Ordering::SeqCst);
        Ok(
            self.envelopes[&data_binding_requirement_key(&b.node_id, &b.input_name)]
                .predict_cohort
                .clone(),
        )
    }
}
fn handle(owner: &str, id: u64, kind: HandleKind) -> HandleRef {
    HandleRef {
        handle: id,
        kind,
        owner_controller: ControllerId::new(owner).unwrap(),
    }
}
struct Controller {
    id: ControllerId,
    targets: BTreeMap<NodeId, Vec<String>>,
    samples: Vec<SampleId>,
    bad: bool,
    counts: Arc<Counts>,
}
impl RuntimeController for Controller {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }
    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        let ordinal = self.counts.controller.fetch_add(1, Ordering::SeqCst) as u64 + 10;
        let model = task.node_plan.kind == NodeKind::Model;
        let port = if model { "oof" } else { "x_out" };
        let names = self
            .targets
            .get(&task.node_plan.node_id)
            .cloned()
            .unwrap_or_default();
        let mut result: NodeResult = serde_json::from_value(serde_json::json!({
            "node_id":task.node_plan.node_id,
            "outputs":{port:handle(self.id.as_str(),ordinal,if model {HandleKind::Prediction}else{HandleKind::Data})},
            "lineage":{
                "record_id":format!("lineage:synthetic:{ordinal}"),"run_id":task.run_id,
                "node_id":task.node_plan.node_id,"phase":task.phase,
                "controller_id":self.id,"controller_version":task.node_plan.controller_version,
                "variant_id":task.variant_id,"fold_id":task.fold_id,"branch_path":task.branch_path,
                "input_lineage":[],"artifact_refs":[],"params_fingerprint":task.node_plan.params_fingerprint,
                "data_model_shape_fingerprint":null,"aggregation_policy_fingerprint":null,
                "seed":task.seed,"unsafe_flags":[],"metrics":{},"loss_attestations":[],"early_stopping_records":[]
            }
        }))?;
        if model && task.phase == Phase::Predict {
            assert!(!task.artifact_inputs.is_empty());
            result.predictions.push(PredictionBlock {
                prediction_id: Some(format!("prediction:synthetic:{ordinal}")),
                producer_node: task.node_plan.node_id.clone(),
                producer_port: Some(port.into()),
                partition: PredictionPartition::Final,
                fold_id: None,
                sample_ids: self.samples.clone(),
                values: vec![vec![0.0; names.len()]; self.samples.len()],
                target_names: if self.bad {
                    (0..names.len()).map(|i| format!("wrong-{i}")).collect()
                } else {
                    names.clone()
                },
            });
        }
        if model && task.phase == Phase::Explain {
            result.explanations.push(ExplanationBlock {
                producer_node: task.node_plan.node_id.clone(),
                producer_port: Some(port.into()),
                method: "synthetic_zero_importance".into(),
                target_name: Some(names[0].clone()),
                payload: serde_json::json!({"importance":0.0}),
            });
        }
        Ok(result)
    }
    fn hydrate_artifact_payload(
        &self,
        _: &ArtifactMaterializationRequest,
        bytes: &[u8],
    ) -> Result<HandleRef> {
        assert_eq!(bytes, b"synthetic-host-fitted-state-v1");
        let id = self.counts.hydrate.fetch_add(1, Ordering::SeqCst) as u64 + 100;
        Ok(handle(self.id.as_str(), id, HandleKind::Artifact))
    }
    fn release_hydrated_artifact_payload(&self, _: &HandleRef) -> Result<()> {
        self.counts.release.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}
pub fn registry(
    package: &PortablePredictorPackage,
    envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
    counts: &Arc<Counts>,
    bad: bool,
) -> RuntimeControllerRegistry {
    let targets = package
        .output_bindings
        .iter()
        .map(|b| (b.node_id.clone(), b.target_names.clone()))
        .collect::<BTreeMap<_, _>>();
    let samples = envelopes
        .values()
        .next()
        .unwrap()
        .coordinator_relations
        .as_ref()
        .unwrap()
        .records
        .iter()
        .map(|r| r.sample_id.clone())
        .collect::<Vec<_>>();
    let mut registry = RuntimeControllerRegistry::new();
    for id in package.effective_plan.controller_manifests.keys() {
        registry
            .register(Box::new(Controller {
                id: id.clone(),
                targets: targets.clone(),
                samples: samples.clone(),
                bad,
                counts: counts.clone(),
            }))
            .unwrap();
    }
    registry
}
