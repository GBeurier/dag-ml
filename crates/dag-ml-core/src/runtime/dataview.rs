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
    fn coordinator_relations(&self, _binding: &DataBinding) -> Result<Option<SampleRelationSet>> {
        Ok(None)
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
