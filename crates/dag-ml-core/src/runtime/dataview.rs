// Auto-split from the former monolithic `runtime.rs` (pure refactor).
use super::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataMaterializationRequest {
    pub run_id: RunId,
    pub node_id: NodeId,
    pub input_name: String,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub fold_id: Option<FoldId>,
    pub binding: crate::data::DataBinding,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataProviderViewSpec {
    #[serde(default)]
    pub sample_ids: Option<Vec<SampleId>>,
    pub partition: DataRequestPartition,
    #[serde(default)]
    pub fold_id: Option<FoldId>,
    #[serde(default)]
    pub source_ids: Option<Vec<String>>,
    #[serde(default)]
    pub columns: Option<Vec<String>>,
    pub include_augmented: bool,
    pub include_excluded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_view: Option<crate::data::BranchViewPlan>,
    #[serde(default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

pub const DATA_OUTPUT_PROVENANCE_KEY: &str = "dag_ml_output";
pub const DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION: u32 = 1;
pub const DATA_OUTPUT_PROVENANCE_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/data_output_provenance.v1.schema.json";
pub const NODE_TASK_SCHEMA_VERSION: u32 = 1;
pub const NODE_TASK_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/node_task.v1.schema.json";
pub const NODE_RESULT_SCHEMA_VERSION: u32 = 1;
pub const NODE_RESULT_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/node_result.v1.schema.json";

pub(crate) fn default_data_output_provenance_schema_version() -> u32 {
    DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION
}

impl DataProviderViewSpec {
    pub fn validate(&self) -> Result<()> {
        validate_optional_ids("sample id", &self.sample_ids)?;
        validate_optional_strings("source id", &self.source_ids)?;
        validate_optional_strings("column", &self.columns)?;
        match self.partition {
            DataRequestPartition::FoldTrain | DataRequestPartition::FoldValidation => {
                if self.sample_ids.is_some() && self.fold_id.is_none() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "data provider view {:?} with explicit sample ids requires a fold id",
                        self.partition
                    )));
                }
            }
            DataRequestPartition::FullTrain | DataRequestPartition::Predict => {
                if self.fold_id.is_some() {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "data provider view {:?} must not carry a fold id",
                        self.partition
                    )));
                }
            }
        }
        for key in self.extra.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::RuntimeValidation(
                    "data provider view extra contains an empty key".to_string(),
                ));
            }
        }
        if let Some(branch_view) = &self.branch_view {
            branch_view.validate()?;
        }
        self.output_provenance()?;
        Ok(())
    }

    pub fn output_provenance(&self) -> Result<Option<DataOutputProvenance>> {
        let Some(value) = self.extra.get(DATA_OUTPUT_PROVENANCE_KEY) else {
            return Ok(None);
        };
        let provenance: DataOutputProvenance = serde_json::from_value(value.clone())?;
        provenance.validate()?;
        Ok(Some(provenance))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataOutputProvenance {
    #[serde(default = "default_data_output_provenance_schema_version")]
    pub schema_version: u32,
    pub producer_node: NodeId,
    pub producer_port: String,
    pub producer_phase: Phase,
    #[serde(default)]
    pub variant_id: Option<VariantId>,
    #[serde(default)]
    pub fold_id: Option<FoldId>,
    #[serde(default)]
    pub shape_plan_fingerprint: Option<String>,
    #[serde(default)]
    pub aggregation_policy_fingerprint: Option<String>,
    #[serde(default)]
    pub feature_namespace: Option<String>,
    #[serde(default)]
    pub feature_schema_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation_plan: Option<RepresentationPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation_replay_manifest: Option<RepresentationReplayManifest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation_compatibility: Option<RepresentationCompatibilityReport>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relation_delta_fingerprint: Option<String>,
    #[serde(default)]
    pub shape_deltas: Vec<ShapeDelta>,
}

impl DataOutputProvenance {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION {
            return Err(DagMlError::RuntimeValidation(format!(
                "data output provenance for `{}` uses unsupported schema_version {}, expected {}",
                self.producer_node, self.schema_version, DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION
            )));
        }
        if self.producer_port.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "data output provenance for `{}` has empty producer_port",
                self.producer_node
            )));
        }
        validate_optional_fingerprint(
            "shape_plan_fingerprint",
            &self.shape_plan_fingerprint,
            &self.producer_node,
        )?;
        validate_optional_fingerprint(
            "aggregation_policy_fingerprint",
            &self.aggregation_policy_fingerprint,
            &self.producer_node,
        )?;
        validate_optional_fingerprint(
            "feature_schema_fingerprint",
            &self.feature_schema_fingerprint,
            &self.producer_node,
        )?;
        validate_optional_fingerprint(
            "relation_delta_fingerprint",
            &self.relation_delta_fingerprint,
            &self.producer_node,
        )?;
        if let Some(representation_plan) = &self.representation_plan {
            representation_plan.validate().map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "data output provenance for `{}` has invalid representation_plan: {error}",
                    self.producer_node
                ))
            })?;
        }
        if let Some(replay_manifest) = &self.representation_replay_manifest {
            replay_manifest.validate().map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "data output provenance for `{}` has invalid representation_replay_manifest: {error}",
                    self.producer_node
                ))
            })?;
        }
        if let Some(report) = &self.representation_compatibility {
            report.validate().map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "data output provenance for `{}` has invalid representation_compatibility: {error}",
                    self.producer_node
                ))
            })?;
        }
        if self
            .feature_namespace
            .as_ref()
            .is_some_and(|namespace| namespace.trim().is_empty())
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "data output provenance for `{}` has empty feature_namespace",
                self.producer_node
            )));
        }
        for delta in &self.shape_deltas {
            delta.validate()?;
            if delta.node_id != self.producer_node {
                return Err(DagMlError::RuntimeValidation(format!(
                    "data output provenance for `{}` contains shape delta for `{}`",
                    self.producer_node, delta.node_id
                )));
            }
        }
        if let Some(feature_schema_fingerprint) = &self.feature_schema_fingerprint {
            if let Some(last_feature_delta) = self
                .shape_deltas
                .iter()
                .rev()
                .find(|delta| delta.kind == ShapeDeltaKind::Feature)
            {
                if &last_feature_delta.after_fingerprint != feature_schema_fingerprint {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "data output provenance for `{}` has feature_schema_fingerprint `{feature_schema_fingerprint}` but last feature delta ends at `{}`",
                        self.producer_node, last_feature_delta.after_fingerprint
                    )));
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_optional_fingerprint(
    label: &str,
    fingerprint: &Option<String>,
    producer_node: &NodeId,
) -> Result<()> {
    let Some(fingerprint) = fingerprint else {
        return Ok(());
    };
    if fingerprint.len() != 64 || !fingerprint.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DagMlError::RuntimeValidation(format!(
            "data output provenance for `{producer_node}` has invalid {label}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_optional_ids<T>(label: &str, values: &Option<Vec<T>>) -> Result<()>
where
    T: Ord + ToString,
{
    let Some(values) = values else {
        return Ok(());
    };
    if values.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "data provider view {label} list is empty"
        )));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(DagMlError::RuntimeValidation(format!(
                "data provider view has duplicate {label} `{}`",
                value.to_string()
            )));
        }
    }
    Ok(())
}

pub(crate) fn validate_optional_strings(label: &str, values: &Option<Vec<String>>) -> Result<()> {
    let Some(values) = values else {
        return Ok(());
    };
    if values.is_empty() {
        return Err(DagMlError::RuntimeValidation(format!(
            "data provider view {label} list is empty"
        )));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        if value.trim().is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "data provider view contains an empty {label}"
            )));
        }
        if !seen.insert(value.as_str()) {
            return Err(DagMlError::RuntimeValidation(format!(
                "data provider view has duplicate {label} `{value}`"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataViewRequest {
    pub run_id: RunId,
    pub node_id: NodeId,
    pub input_name: String,
    pub phase: Phase,
    pub variant_id: Option<VariantId>,
    pub fold_id: Option<FoldId>,
    pub binding: crate::data::DataBinding,
    pub data_handle: HandleRef,
    pub view: DataProviderViewSpec,
}

pub trait RuntimeDataProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> Result<HandleRef>;
    fn make_view(&self, request: &DataViewRequest) -> Result<HandleRef>;
    /// Attest the exact feature and target content bound to one training input.
    ///
    /// Legacy phase execution may return `None`; the native W1 training
    /// operation requires `Some` and compares it byte-for-byte with the signed
    /// [`TrainingDataIdentity`](crate::training::TrainingDataIdentity).
    fn training_data_identity(
        &self,
        _binding: &DataBinding,
    ) -> Result<Option<crate::training::TrainingDataIdentity>> {
        Ok(None)
    }
    fn coordinator_relations(&self, _binding: &DataBinding) -> Result<Option<SampleRelationSet>> {
        Ok(None)
    }
}

#[derive(Debug)]
struct EnvelopeAttestation {
    binding: DataBinding,
    envelope: ExternalDataPlanEnvelope,
    identity: crate::training::TrainingDataIdentity,
}

/// Owns a host data provider while supplying exact, envelope-backed training
/// attestations at the runtime trust boundary.
///
/// Construction validates the complete binding/envelope set before the inner
/// provider can be invoked. Runtime calls are delegated only when their full
/// [`DataBinding`] is field-for-field equal to the binding registered for the
/// rendered V1 requirement key.
#[derive(Debug)]
pub struct EnvelopeAttestedRuntimeDataProvider<P> {
    inner: P,
    attestations: BTreeMap<String, EnvelopeAttestation>,
}

impl<P> EnvelopeAttestedRuntimeDataProvider<P> {
    pub fn new<I>(
        inner: P,
        bindings: I,
        mut envelopes: BTreeMap<String, ExternalDataPlanEnvelope>,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = DataBinding>,
    {
        let mut bindings_by_key: BTreeMap<String, DataBinding> = BTreeMap::new();
        for binding in bindings {
            binding.validate()?;
            let key = data_binding_requirement_key(&binding.node_id, &binding.input_name);
            if let Some(previous) = bindings_by_key.get(&key) {
                let detail = if previous.node_id == binding.node_id
                    && previous.input_name == binding.input_name
                {
                    "duplicates the same coordinates"
                } else {
                    "uses distinct coordinates that collide under the V1 node.input spelling"
                };
                return Err(DagMlError::RuntimeValidation(format!(
                    "data binding requirement key `{key}` {detail}"
                )));
            }
            bindings_by_key.insert(key, binding);
        }

        let expected_keys = bindings_by_key.keys().cloned().collect::<BTreeSet<_>>();
        let actual_keys = envelopes.keys().cloned().collect::<BTreeSet<_>>();
        if expected_keys != actual_keys {
            let missing = expected_keys
                .difference(&actual_keys)
                .cloned()
                .collect::<Vec<_>>();
            let unexpected = actual_keys
                .difference(&expected_keys)
                .cloned()
                .collect::<Vec<_>>();
            return Err(DagMlError::RuntimeValidation(format!(
                "attested data envelopes must exactly cover runtime bindings (missing: [{}]; unexpected: [{}])",
                missing.join(", "),
                unexpected.join(", ")
            )));
        }

        let mut attestations = BTreeMap::new();
        for (key, binding) in bindings_by_key {
            let envelope = envelopes
                .remove(&key)
                .expect("exact key coverage was checked above");
            let identity =
                crate::training::TrainingDataIdentity::from_binding_envelope(&binding, &envelope)?;
            attestations.insert(
                key,
                EnvelopeAttestation {
                    binding,
                    envelope,
                    identity,
                },
            );
        }

        Ok(Self {
            inner,
            attestations,
        })
    }

    pub fn inner(&self) -> &P {
        &self.inner
    }

    pub fn into_inner(self) -> P {
        self.inner
    }

    fn attestation_for_binding(&self, binding: &DataBinding) -> Result<&EnvelopeAttestation> {
        binding.validate()?;
        let key = data_binding_requirement_key(&binding.node_id, &binding.input_name);
        let attestation = self.attestations.get(&key).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "runtime data binding `{key}` has no registered envelope attestation"
            ))
        })?;
        if attestation.binding != *binding {
            return Err(DagMlError::RuntimeValidation(format!(
                "runtime data binding `{key}` does not exactly match its attested binding"
            )));
        }
        Ok(attestation)
    }

    fn validate_request_binding(
        &self,
        node_id: &NodeId,
        input_name: &str,
        binding: &DataBinding,
    ) -> Result<()> {
        if node_id != &binding.node_id || input_name != binding.input_name {
            return Err(DagMlError::RuntimeValidation(format!(
                "runtime data request coordinates `{node_id}.{input_name}` do not match binding `{}`",
                data_binding_requirement_key(&binding.node_id, &binding.input_name)
            )));
        }
        self.attestation_for_binding(binding)?;
        Ok(())
    }
}

impl<P: RuntimeDataProvider> RuntimeDataProvider for EnvelopeAttestedRuntimeDataProvider<P> {
    fn materialize(&self, request: &DataMaterializationRequest) -> Result<HandleRef> {
        self.validate_request_binding(&request.node_id, &request.input_name, &request.binding)?;
        self.inner.materialize(request)
    }

    fn make_view(&self, request: &DataViewRequest) -> Result<HandleRef> {
        request.view.validate()?;
        self.validate_request_binding(&request.node_id, &request.input_name, &request.binding)?;
        self.inner.make_view(request)
    }

    fn training_data_identity(
        &self,
        binding: &DataBinding,
    ) -> Result<Option<crate::training::TrainingDataIdentity>> {
        Ok(Some(
            self.attestation_for_binding(binding)?.identity.clone(),
        ))
    }

    fn coordinator_relations(&self, binding: &DataBinding) -> Result<Option<SampleRelationSet>> {
        Ok(self
            .attestation_for_binding(binding)?
            .envelope
            .coordinator_relations
            .clone())
    }
}

pub trait RuntimeController: Send + Sync {
    fn controller_id(&self) -> &ControllerId;
    fn invoke(&self, task: &NodeTask) -> Result<NodeResult>;

    fn invoke_aggregation(
        &self,
        task: &AggregationControllerTask,
    ) -> Result<AggregationControllerResult> {
        Err(DagMlError::RuntimeValidation(format!(
            "runtime controller `{}` does not implement aggregation task `{}`",
            self.controller_id(),
            task.task_id
        )))
    }
}
pub(crate) struct CollectedInputs {
    pub(crate) handles: BTreeMap<String, HandleRef>,
    pub(crate) data_views: BTreeMap<String, DataProviderViewSpec>,
    pub(crate) prediction_inputs: BTreeMap<String, PredictionInputSpec>,
    pub(crate) skip_node: bool,
}

pub(crate) fn data_view_key(input_name: &str) -> String {
    format!("data:{input_name}")
}

pub(crate) fn validation_data_view_key(input_name: &str) -> String {
    format!("{input_name}:validation")
}

pub(crate) fn derive_output_data_views(
    plan: &ExecutionPlan,
    task: &NodeTask,
    result: &NodeResult,
) -> Result<BTreeMap<String, DataProviderViewSpec>> {
    let node = plan
        .graph_plan
        .graph
        .nodes
        .iter()
        .find(|node| node.id == task.node_plan.node_id)
        .expect("execution plan was validated");
    let mut views = BTreeMap::new();
    for port in node
        .ports
        .outputs
        .iter()
        .filter(|port| port.kind == PortKind::Data)
    {
        let Some(handle) = result.outputs.get(&port.name) else {
            continue;
        };
        if !matches!(handle.kind, HandleKind::Data | HandleKind::DataView) {
            return Err(DagMlError::RuntimeValidation(format!(
                "node `{}` emitted data output `{}` with non-data/data-view handle kind {:?}",
                task.node_plan.node_id, port.name, handle.kind
            )));
        }
        if let Some(view) = primary_output_data_view(task) {
            views.insert(
                port.name.clone(),
                output_data_view_for_port(task, result, &port.name, view)?,
            );
        }
        if let Some(validation_view) = validation_output_data_view(task) {
            views.insert(
                validation_data_view_key(&port.name),
                output_data_view_for_port(task, result, &port.name, validation_view)?,
            );
        }
    }
    Ok(views)
}

pub(crate) fn output_data_view_for_port(
    task: &NodeTask,
    result: &NodeResult,
    port_name: &str,
    base_view: &DataProviderViewSpec,
) -> Result<DataProviderViewSpec> {
    let mut view = base_view.clone();
    if let Some(upstream_provenance) = view.extra.remove(DATA_OUTPUT_PROVENANCE_KEY) {
        let provenance: DataOutputProvenance =
            serde_json::from_value(upstream_provenance).map_err(|error| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` cannot propagate data output `{port_name}` because upstream data output provenance is invalid JSON: {error}",
                    task.node_plan.node_id
                ))
            })?;
        provenance.validate().map_err(|error| {
            DagMlError::RuntimeValidation(format!(
                "node `{}` cannot propagate data output `{port_name}` because upstream data output provenance is invalid: {error}",
                task.node_plan.node_id
            ))
        })?;
    }
    let shape_deltas = result
        .shape_deltas
        .iter()
        .filter(|delta| delta.node_id == task.node_plan.node_id)
        .cloned()
        .collect::<Vec<_>>();
    let mut provenance = DataOutputProvenance {
        schema_version: DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION,
        producer_node: task.node_plan.node_id.clone(),
        producer_port: port_name.to_string(),
        producer_phase: task.phase,
        variant_id: task.variant_id.clone(),
        fold_id: task.fold_id.clone(),
        shape_plan_fingerprint: None,
        aggregation_policy_fingerprint: None,
        feature_namespace: None,
        feature_schema_fingerprint: None,
        representation_plan: None,
        representation_replay_manifest: None,
        representation_compatibility: None,
        relation_delta_fingerprint: None,
        shape_deltas,
    };
    if let Some(shape_plan) = &task.node_plan.shape_plan {
        provenance.shape_plan_fingerprint = Some(stable_json_fingerprint(shape_plan)?);
        provenance.aggregation_policy_fingerprint =
            Some(stable_json_fingerprint(&shape_plan.aggregation_policy)?);
        provenance.feature_namespace = shape_plan.feature_namespace.clone();
        provenance.feature_schema_fingerprint =
            output_feature_schema_fingerprint(shape_plan, result);
    }
    provenance.validate()?;

    view.extra.insert(
        DATA_OUTPUT_PROVENANCE_KEY.to_string(),
        serde_json::to_value(provenance)?,
    );
    view.validate()?;
    Ok(view)
}

pub(crate) fn output_feature_schema_fingerprint(
    shape_plan: &crate::policy::DataModelShapePlan,
    result: &NodeResult,
) -> Option<String> {
    result
        .shape_deltas
        .iter()
        .rev()
        .find(|delta| delta.kind == ShapeDeltaKind::Feature)
        .map(|delta| delta.after_fingerprint.clone())
        .or_else(|| shape_plan.feature_schema_fingerprint.clone())
}

pub(crate) fn primary_output_data_view(task: &NodeTask) -> Option<&DataProviderViewSpec> {
    task.data_views
        .values()
        .find(|view| view.partition != DataRequestPartition::FoldValidation)
        .or_else(|| task.data_views.values().next())
}

pub(crate) fn validation_output_data_view(task: &NodeTask) -> Option<&DataProviderViewSpec> {
    task.data_views
        .values()
        .find(|view| view.partition == DataRequestPartition::FoldValidation)
}

pub(crate) fn make_data_view_handle(
    data_provider: &dyn RuntimeDataProvider,
    ctx: &RunContext,
    node_plan: &NodePlan,
    scope: &PhaseScope,
    binding: &DataBinding,
    data_handle: &HandleRef,
    view: &DataProviderViewSpec,
) -> Result<HandleRef> {
    view.validate()?;
    let view_handle = data_provider.make_view(&DataViewRequest {
        run_id: ctx.run_id.clone(),
        node_id: node_plan.node_id.clone(),
        input_name: binding.input_name.clone(),
        phase: scope.phase,
        variant_id: scope.variant_id.clone(),
        fold_id: scope.fold_id.clone(),
        binding: binding.clone(),
        data_handle: data_handle.clone(),
        view: view.clone(),
    })?;
    // A data view is delivered to the controller as a data input, so the
    // provider must return a data-bearing handle. Refuse a model / artifact /
    // prediction / relation handle masquerading as a view across the ABI.
    if !matches!(view_handle.kind, HandleKind::Data | HandleKind::DataView) {
        return Err(DagMlError::RuntimeValidation(format!(
            "node `{}` data view `{}` resolved to a non-data/data-view handle kind {:?}",
            node_plan.node_id, binding.input_name, view_handle.kind
        )));
    }
    Ok(view_handle)
}

pub(crate) fn data_view_for_scope(
    binding: &DataBinding,
    fold_set: Option<&FoldSet>,
    scope: &PhaseScope,
    branch_view: Option<&crate::data::BranchViewPlan>,
    excluded_samples: &BTreeSet<SampleId>,
) -> Result<DataProviderViewSpec> {
    let partition = data_partition_for_scope(binding, scope);
    // During FIT_CV and REFIT this primary view IS the training input; during
    // PREDICT/EXPLAIN (and the planning phases) it is a non-fit read.
    let role = match scope.phase {
        Phase::FitCv | Phase::Refit => DataViewRole::Fit,
        _ => DataViewRole::NonFit,
    };
    data_view_for_partition(
        binding,
        fold_set,
        scope,
        partition,
        branch_view,
        role,
        excluded_samples,
    )
}

pub(crate) fn validation_data_view_for_scope(
    binding: &DataBinding,
    fold_set: Option<&FoldSet>,
    scope: &PhaseScope,
    branch_view: Option<&crate::data::BranchViewPlan>,
    excluded_samples: &BTreeSet<SampleId>,
) -> Result<Option<DataProviderViewSpec>> {
    if scope.phase != Phase::FitCv || scope.fold_id.is_none() {
        return Ok(None);
    }
    let partition = binding.view_policy.predict_partition;
    if partition == data_partition_for_scope(binding, scope) {
        return Ok(None);
    }
    // This is the validation companion read, never the training input.
    data_view_for_partition(
        binding,
        fold_set,
        scope,
        partition,
        branch_view,
        DataViewRole::NonFit,
        excluded_samples,
    )
    .map(Some)
}

#[cfg(test)]
mod envelope_attested_provider_tests {
    use std::cell::Cell;

    use super::*;

    #[derive(Debug, Default)]
    struct ProbeProvider {
        materialize_calls: Cell<usize>,
        make_view_calls: Cell<usize>,
    }

    impl RuntimeDataProvider for ProbeProvider {
        fn materialize(&self, _request: &DataMaterializationRequest) -> Result<HandleRef> {
            self.materialize_calls.set(self.materialize_calls.get() + 1);
            Ok(HandleRef {
                handle: 41,
                kind: HandleKind::Data,
                owner_controller: ControllerId::new("controller:data.probe").unwrap(),
            })
        }

        fn make_view(&self, _request: &DataViewRequest) -> Result<HandleRef> {
            self.make_view_calls.set(self.make_view_calls.get() + 1);
            Ok(HandleRef {
                handle: 42,
                kind: HandleKind::DataView,
                owner_controller: ControllerId::new("controller:data.probe").unwrap(),
            })
        }
    }

    fn complete_envelope() -> ExternalDataPlanEnvelope {
        let mut envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap();
        envelope.data_content_fingerprint = Some("a".repeat(64));
        envelope.target_content_fingerprint = Some("b".repeat(64));
        envelope
    }

    fn binding_for(
        node_id: &str,
        input_name: &str,
        envelope: &ExternalDataPlanEnvelope,
    ) -> DataBinding {
        DataBinding {
            node_id: NodeId::new(node_id).unwrap(),
            input_name: input_name.to_string(),
            request_id: "request:data.probe".to_string(),
            schema_fingerprint: envelope.schema_fingerprint.clone(),
            plan_fingerprint: envelope.plan_fingerprint.clone(),
            relation_fingerprint: envelope.relation_fingerprint.clone(),
            output_representation: "tabular_numeric".to_string(),
            feature_set_id: Some(input_name.to_string()),
            source_ids: vec!["source:probe".to_string()],
            require_relations: true,
            view_policy: Default::default(),
            metadata: BTreeMap::new(),
        }
    }

    fn envelopes_for(
        binding: &DataBinding,
        envelope: ExternalDataPlanEnvelope,
    ) -> BTreeMap<String, ExternalDataPlanEnvelope> {
        BTreeMap::from([(
            data_binding_requirement_key(&binding.node_id, &binding.input_name),
            envelope,
        )])
    }

    fn materialization_request(binding: &DataBinding) -> DataMaterializationRequest {
        DataMaterializationRequest {
            run_id: RunId::new("run:attested.provider").unwrap(),
            node_id: binding.node_id.clone(),
            input_name: binding.input_name.clone(),
            phase: Phase::Refit,
            variant_id: None,
            fold_id: None,
            binding: binding.clone(),
        }
    }

    #[test]
    fn envelope_attested_provider_delegates_and_returns_exact_attestations() {
        let envelope = complete_envelope();
        let binding = binding_for("model:base", "x", &envelope);
        let expected_identity =
            crate::training::TrainingDataIdentity::from_binding_envelope(&binding, &envelope)
                .unwrap();
        let expected_relations = envelope.coordinator_relations.clone();
        let provider = EnvelopeAttestedRuntimeDataProvider::new(
            ProbeProvider::default(),
            vec![binding.clone()],
            envelopes_for(&binding, envelope),
        )
        .unwrap();

        assert_eq!(
            provider.training_data_identity(&binding).unwrap(),
            Some(expected_identity)
        );
        assert_eq!(
            provider.coordinator_relations(&binding).unwrap(),
            expected_relations
        );

        let materialization = materialization_request(&binding);
        let data_handle = provider.materialize(&materialization).unwrap();
        assert_eq!(data_handle.handle, 41);
        let view_handle = provider
            .make_view(&DataViewRequest {
                run_id: materialization.run_id,
                node_id: binding.node_id.clone(),
                input_name: binding.input_name.clone(),
                phase: Phase::Refit,
                variant_id: None,
                fold_id: None,
                binding: binding.clone(),
                data_handle,
                view: DataProviderViewSpec {
                    sample_ids: None,
                    partition: DataRequestPartition::FullTrain,
                    fold_id: None,
                    source_ids: None,
                    columns: None,
                    include_augmented: true,
                    include_excluded: false,
                    branch_view: None,
                    extra: BTreeMap::new(),
                },
            })
            .unwrap();
        assert_eq!(view_handle.handle, 42);
        assert_eq!(provider.inner().materialize_calls.get(), 1);
        assert_eq!(provider.inner().make_view_calls.get(), 1);

        let inner = provider.into_inner();
        assert_eq!(inner.materialize_calls.get(), 1);
        assert_eq!(inner.make_view_calls.get(), 1);
    }

    #[test]
    fn envelope_attested_provider_requires_exact_envelope_coverage() {
        let envelope = complete_envelope();
        let binding = binding_for("model:base", "x", &envelope);

        let missing = EnvelopeAttestedRuntimeDataProvider::new(
            ProbeProvider::default(),
            vec![binding.clone()],
            BTreeMap::new(),
        )
        .unwrap_err();
        assert!(missing.to_string().contains("exactly cover"));
        assert!(missing.to_string().contains("model:base.x"));

        let mut unexpected = envelopes_for(&binding, envelope.clone());
        unexpected.insert("model:other.x".to_string(), envelope);
        let extra = EnvelopeAttestedRuntimeDataProvider::new(
            ProbeProvider::default(),
            vec![binding],
            unexpected,
        )
        .unwrap_err();
        assert!(extra.to_string().contains("exactly cover"));
        assert!(extra.to_string().contains("model:other.x"));
    }

    #[test]
    fn envelope_attested_provider_rejects_rendered_key_collisions() {
        let envelope = complete_envelope();
        let left = binding_for("a.b", "c", &envelope);
        let right = binding_for("a", "b.c", &envelope);
        assert_eq!(
            data_binding_requirement_key(&left.node_id, &left.input_name),
            data_binding_requirement_key(&right.node_id, &right.input_name)
        );

        let error = EnvelopeAttestedRuntimeDataProvider::new(
            ProbeProvider::default(),
            vec![left.clone(), right],
            envelopes_for(&left, envelope),
        )
        .unwrap_err();
        assert!(error.to_string().contains("distinct coordinates"));
        assert!(error.to_string().contains("a.b.c"));
    }

    #[test]
    fn envelope_attested_provider_refuses_unattested_binding_before_delegation() {
        let envelope = complete_envelope();
        let binding = binding_for("model:base", "x", &envelope);
        let provider = EnvelopeAttestedRuntimeDataProvider::new(
            ProbeProvider::default(),
            vec![binding.clone()],
            envelopes_for(&binding, envelope),
        )
        .unwrap();
        let mut changed = binding;
        changed.request_id = "request:data.changed".to_string();

        let error = provider
            .materialize(&materialization_request(&changed))
            .unwrap_err();
        assert!(error.to_string().contains("does not exactly match"));
        assert_eq!(provider.inner().materialize_calls.get(), 0);
    }

    #[test]
    fn envelope_attested_provider_refuses_incomplete_training_envelope() {
        let mut envelope = complete_envelope();
        envelope.data_content_fingerprint = None;
        let binding = binding_for("model:base", "x", &envelope);
        let error = EnvelopeAttestedRuntimeDataProvider::new(
            ProbeProvider::default(),
            vec![binding.clone()],
            envelopes_for(&binding, envelope),
        )
        .unwrap_err();
        assert!(error.to_string().contains("data content fingerprint"));
    }
}
