//! Native adapters for the closed PLS profile and generic pipeline_steps_v1.
//! All numerical operations and RAW-state inspection use the Methods role API.
use crate::hpo::{MethodsPlsController, MethodsRuntime};
use crate::methods_phase_controls::*;
use crate::runtime::*;
use crate::{
    ArtifactId, ControllerId, DagMlError, LineageId, Phase, PredictionLevel, PredictionUnitId,
    Result,
};
use n4m::roles::{Estimator, FitInputs, ParamValue, Params, RolePipeline};
use n4m::{Context, MatrixRef};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Mutex,
};

fn failure(message: impl std::fmt::Display) -> DagMlError {
    DagMlError::RuntimeValidation(format!("native PLS RolePipeline: {message}"))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Payload {
    schema: String,
    node_id: String,
    params_fingerprint: String,
    target_names: Vec<String>,
    steps: Vec<Value>,
    feature_names: Vec<String>,
    states: Vec<Vec<u8>>,
}
impl Payload {
    fn read(bytes: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(bytes).map_err(failure)?;
        crate::canonical::parse_typed_json(text).map_err(failure)?;
        let saved: Self = serde_json::from_str(text)?;
        crate::NodeId::new(&saved.node_id)?;
        let unique = |names: &[String]| {
            !names.is_empty()
                && names.iter().all(|s| !s.is_empty())
                && names
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == names.len()
        };
        if !matches!(
            saved.schema.as_str(),
            "dagml.methods.regression.v1" | "dagml.methods.pipeline_steps.v1"
        ) || !unique(&saved.target_names)
            || !unique(&saved.feature_names)
            || (saved.schema == "dagml.methods.regression.v1"
                && (!matches!(saved.steps.len(), 1 | 3) || saved.steps.len() != saved.states.len()))
            || saved.steps.is_empty()
            || saved.states.is_empty()
            || saved.states.len() > saved.steps.len()
            || saved.states.iter().any(|s| !s.starts_with(b"N4ME"))
            || saved.params_fingerprint.len() != 64
            || !saved
                .params_fingerprint
                .bytes()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        {
            return Err(failure("invalid closed RAW wrapper"));
        }
        if saved.schema == "dagml.methods.pipeline_steps.v1" {
            preflight_pipeline_steps(&saved.steps)?;
            return Ok(saved);
        }
        // Recompute a closed native recipe from saved scalar declarations;
        // native import below separately verifies them against each state.
        let last = saved.steps.last().expect("validated nonempty recipe");
        let params = last
            .get("params")
            .and_then(Value::as_object)
            .ok_or_else(|| failure("missing model params"))?;
        let scale_x = params
            .get("scale_x")
            .and_then(Value::as_bool)
            .ok_or_else(|| failure("missing scale_x"))?;
        if params.get("scale_y").and_then(Value::as_bool) != Some(scale_x) {
            return Err(failure("scale must own both native flags atomically"));
        }
        let pipeline = if saved.steps.len() == 3 {
            Some(NativePlsPipeline {
                schema_version: 1,
                pipeline_type: "n4m.snv_savgol_smooth.v1".into(),
                savgol_window: saved.steps[1]["params"]["window_length"]
                    .as_i64()
                    .ok_or_else(|| failure("missing SG window"))?,
                savgol_poly_degree: saved.steps[1]["params"]["polyorder"]
                    .as_i64()
                    .ok_or_else(|| failure("missing SG degree"))?,
            })
        } else {
            None
        };
        let parsed = NativePlsRoleParams {
            native_profile: METHODS_PLS_ROLE_PROFILE.into(),
            n_components: params
                .get("n_components")
                .and_then(Value::as_i64)
                .ok_or_else(|| failure("missing n_components"))?,
            scale: scale_x,
            pipeline,
            phase_controls: None,
        };
        NativePlsRoleParams::from_params(&serde_json::from_value(serde_json::to_value(&parsed)?)?)?;
        if saved.steps != parsed.recipe(Phase::Refit) {
            return Err(failure("RAW recipe is outside the closed PLS profile"));
        }
        Ok(saved)
    }
}

fn exact_json_integer(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| {
        value
            .as_f64()
            .filter(|v| {
                v.is_finite() && v.fract() == 0.0 && *v >= i64::MIN as f64 && *v < (i64::MAX as f64)
            })
            .map(|v| v as i64)
    })
}

fn native_params(context: &Context, step: &Value) -> Result<Params> {
    let id = step["methodId"]
        .as_str()
        .ok_or_else(|| failure("missing native methodId"))?;
    let info = n4m::roles::method_info(id).map_err(failure)?;
    let mut params = Params::new(context, id).map_err(failure)?;
    for (key, value) in step["params"]
        .as_object()
        .ok_or_else(|| failure("missing native params"))?
    {
        let field = info
            .params
            .iter()
            .find(|p| p.name == *key)
            .ok_or_else(|| failure(format!("{id}.params.{key}: unsupported parameter")))?;
        use n4m::roles::ParamType;
        let typed = match field.param_type {
            ParamType::Int => ParamValue::Int(
                exact_json_integer(value)
                    .ok_or_else(|| failure(format!("{id}.{key}: integer required")))?,
            ),
            ParamType::Double => ParamValue::Double(
                value
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| failure(format!("{id}.{key}: finite number required")))?,
            ),
            ParamType::Bool => ParamValue::Bool(
                value
                    .as_bool()
                    .ok_or_else(|| failure(format!("{id}.{key}: boolean required")))?,
            ),
            ParamType::Enum => ParamValue::Enum(
                value
                    .as_str()
                    .ok_or_else(|| failure(format!("{id}.{key}: enum string required")))?
                    .into(),
            ),
            ParamType::IntArray => ParamValue::IntArray(
                value
                    .as_array()
                    .ok_or_else(|| failure(format!("{id}.{key}: array required")))?
                    .iter()
                    .map(|v| {
                        exact_json_integer(v)
                            .ok_or_else(|| failure(format!("{id}.{key}: integer array required")))
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            ParamType::DoubleArray => ParamValue::DoubleArray(
                value
                    .as_array()
                    .ok_or_else(|| failure(format!("{id}.{key}: array required")))?
                    .iter()
                    .map(|v| {
                        v.as_f64().filter(|v| v.is_finite()).ok_or_else(|| {
                            failure(format!("{id}.{key}: finite numeric array required"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
        };
        params.set(key, &typed).map_err(failure)?;
    }
    params.validate(context).map_err(failure)?;
    Ok(params)
}

/// Base recipe first, then all remaining node parameters on the terminal step.
/// Same precedence as the existing WASM controller contract, independent of PLS.
pub fn pipeline_steps_recipe(params: &BTreeMap<String, Value>) -> Result<Vec<Value>> {
    let mut steps =
        crate::runtime::effective_pipeline_recipe(&json!({"type":"N4mRolePipeline"}), params)?;
    for step in &mut steps {
        let step = step.as_object_mut().expect("owner validated object");
        if let Some(class) = step.remove("class") {
            let id = class
                .as_str()
                .and_then(|id| id.strip_prefix("n4m:"))
                .ok_or_else(|| failure("pipeline_steps_v1 class requires n4m:<method id>"))?;
            step.insert("methodId".into(), json!(id));
        }
        step.entry("params").or_insert_with(|| json!({}));
    }
    Ok(steps)
}

/// Validate every candidate through the live Methods manifest before the first fit.
pub fn preflight_pipeline_steps(steps: &[Value]) -> Result<()> {
    use n4m::roles::{FitInput, InputRequirement, ROLE_REGRESSOR};
    if steps.is_empty() {
        return Err(failure("empty recipe"));
    }
    let context = Context::new().map_err(failure)?;
    for (i, step) in steps.iter().enumerate() {
        let id = step["methodId"]
            .as_str()
            .ok_or_else(|| failure("methodId required"))?;
        let info = n4m::roles::method_info(id).map_err(failure)?;
        if i + 1 == steps.len() && info.roles & ROLE_REGRESSOR == 0 {
            return Err(failure(format!("{id}: terminal role must be regressor")));
        }
        if info.roles & n4m::roles::ROLE_SAMPLE_FILTER == 0
            && (info.capabilities & n4m::roles::CAP_SERIALIZABLE == 0
                || info.capabilities & n4m::roles::CAP_RETAINS_TRAINING_ROWS != 0)
        {
            return Err(failure(format!("{id}: portable pipeline requires serializable state without retained training rows")));
        }
        for input in [
            FitInput::Labels,
            FitInput::SampleWeight,
            FitInput::Groups,
            FitInput::FeatureGroups,
            FitInput::Blocks,
            FitInput::Axis,
            FitInput::TargetDomain,
            FitInput::FoldIds,
        ] {
            if info.input(input) == InputRequirement::Required {
                return Err(failure(format!(
                    "{id}: unsupported required fit input {input:?}"
                )));
            }
        }
        native_params(&context, step)?;
    }
    // Native RolePipeline validates intermediate transformer/selector/filter roles.
    pipeline(&context, steps, &[]).map(|_| ())
}

fn pipeline(context: &Context, steps: &[Value], names: &[String]) -> Result<RolePipeline> {
    let params = steps
        .iter()
        .map(|s| native_params(context, s))
        .collect::<Result<Vec<_>>>()?;
    let declarations = steps
        .iter()
        .zip(&params)
        .map(|(s, p)| (s["methodId"].as_str().expect("validated method"), Some(p)))
        .collect::<Vec<_>>();
    let mut model = RolePipeline::new(context, &declarations).map_err(failure)?;
    let names = names.iter().map(String::as_str).collect::<Vec<_>>();
    if !names.is_empty() {
        model.set_feature_names(context, &names).map_err(failure)?;
    }
    Ok(model)
}

/// Inspect states after official import; PLS flags are validated by Methods
/// against the actual imported sub-model, not inferred from scale arrays.
pub fn inspect_methods_role_pipeline_params(
    bytes: &[u8],
    _runtime: &MethodsRuntime,
) -> Result<Value> {
    let saved = Payload::read(bytes)?;
    let context = Context::new().map_err(failure)?;
    let mut model = pipeline(&context, &saved.steps, &saved.feature_names)?;
    let states = saved.states.iter().map(Vec::as_slice).collect::<Vec<_>>();
    model.import_states(&context, &states).map_err(failure)?;
    if saved.schema == "dagml.methods.pipeline_steps.v1" {
        // Native import checks role/state/recorded-parameter contracts, including
        // train-only filters which have no state_index. Do not zip states to steps.
        return Ok(
            json!({"schema":saved.schema,"steps":saved.steps,"feature_names":saved.feature_names,"target_names":saved.target_names}),
        );
    }
    let mut inspected = Vec::new();
    for (step, state) in saved.steps.iter().zip(&saved.states) {
        let estimator = Estimator::from_n4me(&context, state).map_err(failure)?;
        if estimator.method_id().map_err(failure)?
            != step["methodId"].as_str().expect("validated method")
        {
            return Err(failure("imported method differs from recipe"));
        }
        let actual = estimator.params(&context).map_err(failure)?;
        let expected = native_params(&context, step)?;
        let mut values = serde_json::Map::new();
        for (name, declared) in step["params"].as_object().expect("validated params") {
            let value = if declared.is_f64() {
                let read = actual.double_values(name).map_err(failure)?;
                if read != expected.double_values(name).map_err(failure)? || read.len() != 1 {
                    return Err(failure("imported numeric params differ from recipe"));
                }
                json!(read[0])
            } else {
                let read = actual.int_values(name).map_err(failure)?;
                if read != expected.int_values(name).map_err(failure)? || read.len() != 1 {
                    return Err(failure(
                        "imported integer/bool/enum params differ from recipe",
                    ));
                }
                if declared.is_boolean() {
                    json!(read[0] != 0)
                } else if declared.is_string() {
                    declared.clone()
                } else {
                    json!(read[0])
                }
            };
            values.insert(name.clone(), value);
        }
        inspected.push(json!({"methodId":step["methodId"],"params":values}));
    }
    let last = inspected.last().expect("validated model step");
    Ok(
        json!({"native_profile":METHODS_PLS_ROLE_PROFILE,"node_id":saved.node_id,"params_fingerprint":saved.params_fingerprint,"target_names":saved.target_names,"feature_names":saved.feature_names,"steps":inspected,"model_params":{"n_components":last["params"]["n_components"],"scale":last["params"]["scale_x"]}}),
    )
}

pub struct MethodsNativeRegressionController {
    id: ControllerId,
    runtime: MethodsRuntime,
    next: AtomicU64,
    pipeline_steps_v1: bool,
    exported: Mutex<BTreeMap<ArtifactId, Vec<u8>>>,
    hydrated: Mutex<BTreeMap<u64, Vec<u8>>>,
}
impl MethodsNativeRegressionController {
    pub fn new(runtime: MethodsRuntime) -> Self {
        Self {
            id: ControllerId::new(METHODS_NATIVE_REGRESSION_CONTROLLER)
                .expect("constant controller"),
            runtime,
            pipeline_steps_v1: false,
            next: AtomicU64::new(0),
            exported: Mutex::new(BTreeMap::new()),
            hydrated: Mutex::new(BTreeMap::new()),
        }
    }
    /// Native implementation of the existing pipeline_steps_v1 / 1.1.0 contract.
    pub fn pipeline_steps_v1(runtime: MethodsRuntime) -> Self {
        let mut controller = Self::new(runtime);
        controller.id = ControllerId::new(crate::PORTABLE_STEPS_CONTROLLER).expect("constant ID");
        controller.pipeline_steps_v1 = true;
        controller
    }
    fn version(&self) -> &'static str {
        if self.pipeline_steps_v1 {
            crate::PORTABLE_STEPS_VERSION
        } else {
            METHODS_NATIVE_REGRESSION_VERSION
        }
    }
    fn payload_schema(&self) -> &'static str {
        if self.pipeline_steps_v1 {
            "dagml.methods.pipeline_steps.v1"
        } else {
            "dagml.methods.regression.v1"
        }
    }
    fn handle(&self, kind: HandleKind) -> HandleRef {
        HandleRef {
            handle: self.next.fetch_add(1, Ordering::SeqCst) + 1,
            kind,
            owner_controller: self.id.clone(),
        }
    }
    fn export(
        &self,
        task: &NodeTask,
        context: &Context,
        model: &RolePipeline,
        steps: Vec<Value>,
        data: &MethodsPlsDataset,
        names: Vec<String>,
    ) -> Result<(ArtifactRef, HandleRef)> {
        let saved = Payload {
            schema: self.payload_schema().into(),
            node_id: task.node_plan.node_id.to_string(),
            params_fingerprint: task.node_plan.params_fingerprint.clone(),
            target_names: data.target_names.clone(),
            steps,
            feature_names: names,
            states: model.export_states(context, false).map_err(failure)?,
        };
        let bytes = serde_json::to_vec(&saved)?;
        inspect_methods_role_pipeline_params(&bytes, &self.runtime)?;
        let raw = format!("{:x}", Sha256::digest(&bytes));
        let scope = if task
            .variant_id
            .as_ref()
            .is_some_and(|v| v.as_str().starts_with("hpo:scope:"))
        {
            crate::campaign::stable_json_fingerprint(&(&task.node_plan.node_id, &task.variant_id))?
        } else {
            format!(
                "{}:{}",
                task.node_plan.node_id,
                task.variant_id
                    .as_ref()
                    .map(|v| v.as_str())
                    .unwrap_or("base")
            )
        };
        let artifact = ArtifactRef {
            id: ArtifactId::new(format!("artifact:native_roles:{scope}:refit"))?,
            kind: "methods_role_pipeline".into(),
            controller_id: self.id.clone(),
            backend: Some(ArtifactBackend::Raw),
            uri: Some(format!("artifacts/{raw}.json")),
            content_fingerprint: Some(raw),
            size_bytes: Some(bytes.len() as u64),
            plugin: Some(METHODS_NATIVE_REGRESSION_PLUGIN.into()),
            plugin_version: Some(self.version().into()),
            abi_major: None,
            abi_min_minor: None,
            native_predictor_descriptor: None,
            native_estimator_descriptor: None,
        };
        artifact.validate()?;
        self.exported
            .lock()
            .map_err(failure)?
            .insert(artifact.id.clone(), bytes);
        Ok((artifact, self.handle(HandleKind::Model)))
    }
}
impl RuntimeController for MethodsNativeRegressionController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }
    fn invoke(&self, _task: &NodeTask) -> Result<NodeResult> {
        Err(failure("requires scheduler-owned numeric views"))
    }
    fn invoke_with_data_provider(
        &self,
        task: &NodeTask,
        provider: &dyn RuntimeDataProvider,
    ) -> Result<NodeResult> {
        if task.node_plan.controller_id != self.id
            || task.node_plan.controller_version != self.version()
            || task.node_plan.kind != crate::graph::NodeKind::Model
        {
            return Err(failure("task controller identity mismatch"));
        }
        let steps = if self.pipeline_steps_v1 {
            pipeline_steps_recipe(&task.node_plan.params)?
        } else {
            NativePlsRoleParams::from_params(&task.node_plan.params)?.recipe(task.phase)
        };
        if self.pipeline_steps_v1 {
            preflight_pipeline_steps(&steps)?;
        }
        let request = MethodsPlsController::request(task, provider, "x")?;
        provider.preflight_methods_pls(&request)?;
        let data = provider.methods_pls_data(&request)?;
        data.validate_for(&request)?;
        let context = Context::new().map_err(failure)?;
        let names = (0..data.fit.x.cols)
            .map(|i| format!("feature:{i}"))
            .collect::<Vec<_>>();
        let mut model = pipeline(&context, &steps, &names)?;
        let mut artifact = None;
        match task.phase {
            Phase::FitCv | Phase::Refit => {
                let y = data
                    .fit
                    .y
                    .as_ref()
                    .ok_or_else(|| failure("fit requires target-bound training rows"))?;
                model
                    .fit(
                        &context,
                        &FitInputs::new(matrix(&data.fit.x)?).y(matrix(y)?),
                    )
                    .map_err(failure)?;
                if task.phase == Phase::Refit {
                    artifact = Some(self.export(
                        task,
                        &context,
                        &model,
                        steps.clone(),
                        &data.fit,
                        names.clone(),
                    )?);
                }
            }
            Phase::Predict => {
                let input = task
                    .artifact_inputs
                    .values()
                    .find(|a| {
                        a.controller_id == self.id && a.artifact.kind == "methods_role_pipeline"
                    })
                    .ok_or_else(|| failure("PREDICT requires retained RAW reference"))?;
                let handle = task
                    .input_handles
                    .get(&refit_artifact_input_key(&input.artifact.id))
                    .ok_or_else(|| failure("PREDICT requires fresh hydrated handle"))?;
                if handle.kind != HandleKind::Model || handle.owner_controller != self.id {
                    return Err(failure("PREDICT cannot consume a foreign handle"));
                }
                let bytes = self
                    .hydrated
                    .lock()
                    .map_err(failure)?
                    .remove(&handle.handle)
                    .ok_or_else(|| failure("missing invocation-local hydrated bytes"))?;
                let saved = Payload::read(&bytes)?;
                if saved.schema != self.payload_schema()
                    || saved.node_id != task.node_plan.node_id.as_str()
                    || saved.params_fingerprint != task.node_plan.params_fingerprint
                    || saved.steps != steps
                    || saved.feature_names != names
                    || saved.target_names != data.fit.target_names
                {
                    return Err(failure(
                        "RAW recipe/features/targets differ from current planned task",
                    ));
                }
                model
                    .import_states(
                        &context,
                        &saved.states.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                    )
                    .map_err(failure)?;
            }
            _ => return Err(failure("supports FIT_CV/REFIT/PREDICT only")),
        }
        let mut surfaces = Vec::new();
        if task.phase == Phase::FitCv {
            let validation = data
                .prediction
                .as_ref()
                .ok_or_else(|| failure("FIT_CV requires validation rows"))?;
            surfaces.push((validation.clone(), PredictionPartition::Validation));
            surfaces.push((data.fit.clone(), PredictionPartition::Train));
            surfaces.push((
                crate::methods_estimator::union_rows(&data.fit, validation),
                PredictionPartition::TrainPool,
            ));
        } else {
            surfaces.push((data.fit.clone(), PredictionPartition::Final));
        }
        if task.phase == Phase::Refit {
            if let Some(rows) = &data.prediction {
                surfaces.push((rows.clone(), PredictionPartition::Test));
            }
        }
        let mut predictions = Vec::new();
        let mut targets = Vec::new();
        for (rows, partition) in surfaces {
            model
                .check_features(
                    &context,
                    rows.x.cols,
                    Some(&names.iter().map(String::as_str).collect::<Vec<_>>()),
                )
                .map_err(failure)?;
            let predicted = model
                .predict(&context, matrix(&rows.x)?, None)
                .map_err(failure)?;
            if predicted.rows != rows.sample_ids.len() || predicted.cols != rows.target_names.len()
            {
                return Err(failure(
                    "prediction dimensions do not match named rows/targets",
                ));
            }
            if let Some(y) = &rows.y {
                targets.push(RegressionTargetBlock {
                    validity_masks: None,
                    level: PredictionLevel::Sample,
                    unit_ids: rows
                        .sample_ids
                        .iter()
                        .cloned()
                        .map(PredictionUnitId::Sample)
                        .collect(),
                    values: y.values.chunks(y.cols).map(<[f64]>::to_vec).collect(),
                    target_names: rows.target_names.clone(),
                });
            }
            predictions.push(PredictionBlock {
                prediction_id: Some(
                    if task
                        .variant_id
                        .as_ref()
                        .is_some_and(|v| v.as_str().starts_with("hpo:scope:"))
                    {
                        format!(
                            "native_roles:{}",
                            crate::campaign::stable_json_fingerprint(&(
                                &task.node_plan.node_id,
                                task.phase,
                                &task.variant_id,
                                &task.fold_id,
                                &partition,
                            ))?
                        )
                    } else {
                        format!(
                            "native_roles:{}:{}:{}:{}:{partition:?}",
                            task.node_plan.node_id,
                            task.phase.as_str(),
                            task.variant_id
                                .as_ref()
                                .map(|v| v.as_str())
                                .unwrap_or("base"),
                            task.fold_id.as_ref().map(|f| f.as_str()).unwrap_or("full")
                        )
                    },
                ),
                producer_node: task.node_plan.node_id.clone(),
                producer_port: Some("oof".into()),
                partition,
                fold_id: if task.phase == Phase::FitCv {
                    task.fold_id.clone()
                } else {
                    None
                },
                sample_ids: rows.sample_ids,
                values: predicted
                    .data
                    .chunks(predicted.cols)
                    .map(<[f64]>::to_vec)
                    .collect(),
                target_names: rows.target_names,
            });
        }
        let (artifacts, artifact_handles) = artifact
            .map(|(a, h)| (vec![a.clone()], BTreeMap::from([(a.id, h)])))
            .unwrap_or_default();
        let lineage = LineageRecord {
            record_id: LineageId::new(
                if task
                    .variant_id
                    .as_ref()
                    .is_some_and(|v| v.as_str().starts_with("hpo:scope:"))
                {
                    format!(
                        "lineage:native_roles:{}",
                        crate::campaign::stable_json_fingerprint(&(
                            &task.node_plan.node_id,
                            task.phase,
                            &task.variant_id,
                            &task.fold_id,
                        ))?
                    )
                } else {
                    format!(
                        "lineage:native_roles:{}:{}:{}:{}",
                        task.node_plan.node_id,
                        task.phase.as_str(),
                        task.variant_id
                            .as_ref()
                            .map(|v| v.as_str())
                            .unwrap_or("base"),
                        task.fold_id.as_ref().map(|f| f.as_str()).unwrap_or("full")
                    )
                },
            )?,
            run_id: task.run_id.clone(),
            node_id: task.node_plan.node_id.clone(),
            phase: task.phase,
            controller_id: self.id.clone(),
            controller_version: task.node_plan.controller_version.clone(),
            variant_id: task.variant_id.clone(),
            fold_id: task.fold_id.clone(),
            branch_path: task.branch_path.clone(),
            input_lineage: Vec::new(),
            artifact_refs: artifacts.clone(),
            params_fingerprint: task.node_plan.params_fingerprint.clone(),
            data_model_shape_fingerprint: None,
            aggregation_policy_fingerprint: None,
            seed: task.seed,
            unsafe_flags: Default::default(),
            metrics: Default::default(),
            loss_attestations: Vec::new(),
            early_stopping_records: Vec::new(),
            target_domain: None,
        };
        Ok(NodeResult {
            schema_version: None,
            classification_probabilities: Vec::new(),
            consumed_data_views: BTreeMap::new(),
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([("oof".into(), self.handle(HandleKind::Prediction))]),
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts,
            artifact_handles,
            fit_influence_diagnostics: Vec::new(),
            regression_targets: targets,
            lineage,
        })
    }
    fn export_artifact_payload(&self, id: &ArtifactId) -> Result<Option<Vec<u8>>> {
        Ok(self.exported.lock().map_err(failure)?.remove(id))
    }
    fn hydrate_artifact_payload(
        &self,
        request: &ArtifactMaterializationRequest,
        payload: &[u8],
    ) -> Result<HandleRef> {
        let a = &request.artifact;
        a.validate()?;
        let raw = format!("{:x}", Sha256::digest(payload));
        if request.controller_id != self.id
            || a.controller_id != self.id
            || a.kind != "methods_role_pipeline"
            || a.backend != Some(ArtifactBackend::Raw)
            || a.plugin.as_deref() != Some(METHODS_NATIVE_REGRESSION_PLUGIN)
            || a.plugin_version.as_deref() != Some(self.version())
            || a.abi_major.is_some()
            || a.abi_min_minor.is_some()
            || a.native_predictor_descriptor.is_some()
            || a.native_estimator_descriptor.is_some()
            || a.content_fingerprint.as_deref() != Some(raw.as_str())
            || a.size_bytes != Some(payload.len() as u64)
            || a.uri.as_deref() != Some(format!("artifacts/{raw}.json").as_str())
        {
            return Err(failure("RAW identity/hash/relative path mismatch"));
        }
        let saved = Payload::read(payload)?;
        if saved.schema != self.payload_schema()
            || saved.node_id != request.node_id.as_str()
            || saved.params_fingerprint != request.params_fingerprint
        {
            return Err(failure("RAW node/parameter binding mismatch"));
        }
        inspect_methods_role_pipeline_params(payload, &self.runtime)?;
        let handle = self.handle(HandleKind::Model);
        self.hydrated
            .lock()
            .map_err(failure)?
            .insert(handle.handle, payload.to_vec());
        Ok(handle)
    }
    fn release_hydrated_artifact_payload(&self, handle: &HandleRef) -> Result<()> {
        if handle.kind != HandleKind::Model || handle.owner_controller != self.id {
            return Err(failure("cannot release foreign handle"));
        }
        self.hydrated
            .lock()
            .map_err(failure)?
            .remove(&handle.handle);
        Ok(())
    }
}
fn matrix(m: &MethodsPlsMatrix) -> Result<MatrixRef<'_>> {
    MatrixRef::row_major(&m.values, m.rows, m.cols).map_err(failure)
}

#[cfg(test)]
mod payload_tests {
    use super::*;

    const RAW: &[u8] = include_bytes!(
        "../tests/fixtures/artifact_facets/native_role/source_role_pipeline_0.raw.json"
    );

    #[test]
    fn long_names_preserve_the_genuine_state_and_closed_recipe() {
        let mut saved = Payload::read(RAW).unwrap();
        saved.target_names[0] = "target:".to_owned() + &"t".repeat(4097);
        saved.feature_names[0] = "feature:".to_owned() + &"f".repeat(4097);
        let bytes = serde_json::to_vec(&saved).unwrap();
        assert_eq!(Payload::read(&bytes).unwrap(), saved);
        assert_eq!(saved.states, Payload::read(RAW).unwrap().states);
        saved.feature_names[1] = saved.feature_names[0].clone();
        assert!(Payload::read(&serde_json::to_vec(&saved).unwrap()).is_err());
    }

    #[test]
    #[ignore = "explicit 128 MiB materialized-input regression; no native runtime required"]
    fn wrapper_above_128_mib_preserves_the_genuine_payload() {
        // Whitespace changes only the envelope length, never the fitted wire.
        let mut bytes = Vec::with_capacity(128 * 1024 * 1024 + 1);
        bytes.extend_from_slice(RAW);
        bytes.resize(128 * 1024 * 1024 + 1, b' ');
        assert_eq!(Payload::read(&bytes).unwrap(), Payload::read(RAW).unwrap());
    }
}

#[cfg(all(test, feature = "methods-optimizer-local"))]
mod tests {
    use super::*;
    use crate::{BundleId, NodeId, RunId};
    fn runtime() -> MethodsRuntime {
        MethodsRuntime::configure(
            std::env::var_os("N4M_LIBRARY_PATH")
                .expect("explicit Methods native test runtime required"),
        )
        .unwrap()
    }
    fn witness(scale: bool, with_sg: bool) -> (Payload, Vec<f64>) {
        let rt = runtime();
        let ctx = Context::new().unwrap();
        let params = NativePlsRoleParams {
            native_profile: METHODS_PLS_ROLE_PROFILE.into(),
            n_components: 1,
            scale,
            pipeline: with_sg.then(|| NativePlsPipeline {
                schema_version: 1,
                pipeline_type: "n4m.snv_savgol_smooth.v1".into(),
                savgol_window: 5,
                savgol_poly_degree: 2,
            }),
            phase_controls: None,
        };
        let steps = params.recipe(Phase::Refit);
        let names = (0..9).map(|i| format!("feature:{i}")).collect::<Vec<_>>();
        let x = (0..24)
            .flat_map(|r| {
                (0..9).map(move |c| {
                    ((r * (c + 2) + c * c + 3) % 31) as f64 * (c + 1) as f64
                        + 0.13 * (r * r + c) as f64
                })
            })
            .collect::<Vec<_>>();
        let y = (0..24)
            .map(|r| 0.7 * x[r * 9] + 0.01 * x[r * 9 + 8] + (r % 3) as f64)
            .collect::<Vec<_>>();
        let mut model = pipeline(&ctx, &steps, &names).unwrap();
        model
            .fit(
                &ctx,
                &FitInputs::new(MatrixRef::row_major(&x, 24, 9).unwrap())
                    .y(MatrixRef::row_major(&y, 24, 1).unwrap()),
            )
            .unwrap();
        let predicted = model
            .predict(&ctx, MatrixRef::row_major(&x, 24, 9).unwrap(), None)
            .unwrap()
            .data;
        let saved = Payload {
            schema: "dagml.methods.regression.v1".into(),
            node_id: "model:pls".into(),
            params_fingerprint: "a".repeat(64),
            target_names: vec!["y".into()],
            steps,
            feature_names: names,
            states: model.export_states(&ctx, false).unwrap(),
        };
        inspect_methods_role_pipeline_params(&serde_json::to_vec(&saved).unwrap(), &rt).unwrap();
        (saved, predicted)
    }
    #[test]
    fn native_scale_changes_predictions_and_imported_flags_are_real() {
        let (scaled, a) = witness(true, false);
        let (unscaled, b) = witness(false, false);
        assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6));
        for (saved, expected) in [(scaled, true), (unscaled, false)] {
            let report = inspect_methods_role_pipeline_params(
                &serde_json::to_vec(&saved).unwrap(),
                &runtime(),
            )
            .unwrap();
            assert_eq!(report["model_params"]["scale"], json!(expected));
            assert_eq!(report["model_params"]["n_components"], json!(1));
            let ctx = Context::new().unwrap();
            let mut restored = pipeline(&ctx, &saved.steps, &saved.feature_names).unwrap();
            restored
                .import_states(
                    &ctx,
                    &saved.states.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                )
                .unwrap();
            assert!(restored.is_fitted().unwrap());
        }
    }
    #[test]
    fn native_snv_sg_states_import_without_fit_and_refuse_resealed_scale_claim() {
        let (mut saved, _) = witness(true, true);
        assert_eq!(saved.states.len(), 3);
        assert_eq!(saved.steps[1]["params"]["mode"], json!("interp"));
        inspect_methods_role_pipeline_params(&serde_json::to_vec(&saved).unwrap(), &runtime())
            .unwrap();
        saved.steps[2]["params"]["scale_x"] = json!(false);
        saved.steps[2]["params"]["scale_y"] = json!(false);
        assert!(inspect_methods_role_pipeline_params(
            &serde_json::to_vec(&saved).unwrap(),
            &runtime()
        )
        .is_err());
    }
    #[test]
    fn generic_ridge_pipeline_states_replay_without_fit_and_refuse_changed_alpha() {
        let rt = runtime();
        let context = Context::new().unwrap();
        let steps = vec![
            json!({"methodId":"preprocessing.scatter.msc","params":{}}),
            json!({"methodId":"models.regularized.ridge","params":{"alpha":0.5}}),
        ];
        preflight_pipeline_steps(&steps).unwrap();
        let names = (0..4).map(|i| format!("feature:{i}")).collect::<Vec<_>>();
        let x = (0..20)
            .flat_map(|r| {
                (0..4).map(move |c| {
                    1.0 + 0.3 * r as f64 + (c * c + 1) as f64 * (1.0 + 0.02 * (r % 3) as f64)
                })
            })
            .collect::<Vec<_>>();
        let y = (0..20)
            .map(|r| 0.2 * r as f64 + (r % 3) as f64)
            .collect::<Vec<_>>();
        let mut fitted = pipeline(&context, &steps, &names).unwrap();
        fitted
            .fit(
                &context,
                &FitInputs::new(MatrixRef::row_major(&x, 20, 4).unwrap())
                    .y(MatrixRef::row_major(&y, 20, 1).unwrap()),
            )
            .unwrap();
        let expected = fitted
            .predict(&context, MatrixRef::row_major(&x, 20, 4).unwrap(), None)
            .unwrap()
            .data;
        let mut saved = Payload {
            schema: "dagml.methods.pipeline_steps.v1".into(),
            node_id: "model:ridge".into(),
            params_fingerprint: "a".repeat(64),
            target_names: vec!["y".into()],
            steps,
            feature_names: names,
            states: fitted.export_states(&context, false).unwrap(),
        };
        inspect_methods_role_pipeline_params(&serde_json::to_vec(&saved).unwrap(), &rt).unwrap();
        let mut restored = pipeline(&context, &saved.steps, &saved.feature_names).unwrap();
        restored
            .import_states(
                &context,
                &saved.states.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            )
            .unwrap();
        let actual = restored
            .predict(&context, MatrixRef::row_major(&x, 20, 4).unwrap(), None)
            .unwrap()
            .data;
        assert_eq!(actual, expected);
        saved.steps[1]["params"]["alpha"] = json!(0.75);
        assert!(
            inspect_methods_role_pipeline_params(&serde_json::to_vec(&saved).unwrap(), &rt)
                .is_err()
        );
    }
    #[test]
    fn hydration_hash_recipe_rejection_and_idempotent_release_preserve_lifecycle() {
        let rt = runtime();
        let controller = MethodsNativeRegressionController::new(rt);
        let (saved, _) = witness(true, false);
        let bytes = serde_json::to_vec(&saved).unwrap();
        let raw = format!("{:x}", Sha256::digest(&bytes));
        let mut request = ArtifactMaterializationRequest {
            run_id: RunId::new("run:replay").unwrap(),
            bundle_id: BundleId::new("bundle:replay").unwrap(),
            node_id: NodeId::new("model:pls").unwrap(),
            phase: Phase::Predict,
            variant_id: None,
            controller_id: controller.id.clone(),
            artifact: ArtifactRef {
                id: ArtifactId::new("artifact:pls").unwrap(),
                kind: "methods_role_pipeline".into(),
                controller_id: controller.id.clone(),
                backend: Some(ArtifactBackend::Raw),
                uri: Some(format!("artifacts/{raw}.json")),
                content_fingerprint: Some(raw),
                size_bytes: Some(bytes.len() as u64),
                plugin: Some(METHODS_NATIVE_REGRESSION_PLUGIN.into()),
                plugin_version: Some(METHODS_NATIVE_REGRESSION_VERSION.into()),
                abi_major: None,
                abi_min_minor: None,
                native_predictor_descriptor: None,
                native_estimator_descriptor: None,
            },
            params_fingerprint: saved.params_fingerprint,
            training_loss_fingerprint: None,
        };
        let handle = controller
            .hydrate_artifact_payload(&request, &bytes)
            .unwrap();
        assert_eq!(controller.hydrated.lock().unwrap().len(), 1);
        controller
            .release_hydrated_artifact_payload(&handle)
            .unwrap();
        controller
            .release_hydrated_artifact_payload(&handle)
            .unwrap();
        assert!(controller.hydrated.lock().unwrap().is_empty());
        request.params_fingerprint = "b".repeat(64);
        assert!(controller
            .hydrate_artifact_payload(&request, &bytes)
            .is_err());
        assert!(controller.hydrated.lock().unwrap().is_empty());
    }
}

#[cfg(test)]
mod pipeline_steps_contract_tests {
    use super::*;
    #[test]
    fn terminal_overlays_preserve_base_transform_and_model_flags() {
        let params = BTreeMap::from([
            (
                "pipeline_steps_v1".into(),
                json!([{"class":"n4m:preprocessing.scatter.snv","params":{"with_std":false}},{"methodId":"models.pls.pls_regression","params":{"n_components":5,"scale_x":false,"scale_y":false}}]),
            ),
            ("n_components".into(), json!(2)),
        ]);
        let recipe = pipeline_steps_recipe(&params).unwrap();
        assert_eq!(recipe[0]["methodId"], "preprocessing.scatter.snv");
        assert_eq!(recipe[0]["params"]["with_std"], false);
        assert_eq!(recipe[1]["params"]["n_components"], 2);
        assert_eq!(recipe[1]["params"]["scale_x"], false);
        assert!(recipe[1]["params"].get("pipeline_steps_v1").is_none());
    }
    #[test]
    fn reserved_recipe_structure_fails_without_loading_methods() {
        let params = BTreeMap::from([(
            "pipeline_steps_v1".into(),
            json!([{"methodId":"models.pls.pls_regression","params":{"pipeline_steps_v1":[]}}]),
        )]);
        assert!(pipeline_steps_recipe(&params).is_err());
    }
}
