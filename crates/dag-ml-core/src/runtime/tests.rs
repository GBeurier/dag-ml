use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
};

use super::*;
use crate::aggregation::{aggregate_observation_predictions, ObservationPredictionBlock};
use crate::bundle::{
    build_aggregated_prediction_cache_payload, build_aggregated_prediction_cache_record,
    build_execution_bundle, build_execution_bundle_with_prediction_contracts,
    build_prediction_cache_payload, build_prediction_cache_record, BundlePredictionCachePayloadSet,
    BundlePredictionRequirement, RefitArtifactRecord, ReplayPhaseRequest,
    PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
};
use crate::controller::{
    ArtifactPolicy, ControllerCapability, ControllerFitScope, ControllerManifest,
    ControllerRegistry, RngPolicy,
};
use crate::data::{DataViewPolicy, ExternalDataPlanEnvelope, InMemoryDataProvider};
use crate::fold::{FoldAssignment, FoldSet};
use crate::generation::{
    GenerationChoice, GenerationDimension, GenerationSpec, GenerationStrategy,
};
use crate::graph::{
    EdgeContract, EdgeSpec, GraphInterface, GraphSpec, NodeKind, NodeSpec, PortCardinality,
    PortKind, PortRef, PortSchema, PortSpec,
};
use crate::ids::{
    ArtifactId, ControllerId, FoldId, GroupId, NodeId, ObservationId, SampleId, TargetId,
};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::plan::{build_execution_plan, CampaignSpec, SplitInvocation};
use crate::policy::{
    AggregationControllerSpec, AggregationMethod, AggregationPolicy, DataModelShapePlan,
    FitBoundary, Granularity, LeakageUnitPolicy, ShapeDelta, ShapeDeltaKind, SplitUnit,
};
use crate::relation::{SampleRelation, SampleRelationSet};
use serde_json::json;

struct MockController {
    id: ControllerId,
    handle: u64,
    emit_prediction: bool,
}

struct VariantProbeController {
    id: ControllerId,
    handle: u64,
    variants: Arc<Mutex<Vec<Option<VariantExecutionSpec>>>>,
    node_plans: Arc<Mutex<Vec<NodePlan>>>,
}

impl RuntimeController for VariantProbeController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        self.variants.lock().unwrap().push(task.variant.clone());
        self.node_plans.lock().unwrap().push(task.node_plan.clone());
        let variant_label = task
            .variant_id
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| "base".to_string());
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                "out".to_string(),
                HandleRef {
                    handle: self.handle,
                    kind: HandleKind::Data,
                    owner_controller: self.id.clone(),
                },
            )]),
            predictions: Vec::new(),
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:{}:{:?}:{variant_label}",
                    task.node_plan.node_id, task.phase
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

struct ShapeDataController {
    id: ControllerId,
    handle: u64,
    before_feature_schema: String,
    after_feature_schema: String,
}

impl RuntimeController for ShapeDataController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        let shape_plan = task.node_plan.shape_plan.as_ref().ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "shape data controller `{}` expected a shape plan",
                task.node_plan.node_id
            ))
        })?;
        let output = HandleRef {
            handle: self.handle,
            kind: HandleKind::Data,
            owner_controller: self.id.clone(),
        };
        let shape_delta = ShapeDelta {
            node_id: task.node_plan.node_id.clone(),
            kind: ShapeDeltaKind::Feature,
            before_fingerprint: self.before_feature_schema.clone(),
            after_fingerprint: self.after_feature_schema.clone(),
            metadata: BTreeMap::from([(
                "feature_namespace".to_string(),
                serde_json::Value::String("augmented.noise".to_string()),
            )]),
        };
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([("x_out".to_string(), output)]),
            predictions: Vec::new(),
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            shape_deltas: vec![shape_delta],
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:{}:{:?}:{}:shape",
                    task.node_plan.node_id,
                    task.phase,
                    task.fold_id
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "nofold".to_string())
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: Some(stable_json_fingerprint(shape_plan)?),
                aggregation_policy_fingerprint: Some(stable_json_fingerprint(
                    &shape_plan.aggregation_policy,
                )?),
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

struct DataViewProbeController {
    id: ControllerId,
    observed_views: Arc<Mutex<Vec<BTreeMap<String, DataProviderViewSpec>>>>,
    prediction_sample_ids: Option<Vec<SampleId>>,
}

impl RuntimeController for DataViewProbeController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        self.observed_views
            .lock()
            .unwrap()
            .push(task.data_views.clone());
        let prediction_sample_ids = self.prediction_sample_ids.clone().unwrap_or_else(|| {
            validation_view_sample_ids(task)
                .map(|ids| ids.into_iter().collect::<Vec<_>>())
                .unwrap_or_else(|| vec![SampleId::new("s1").unwrap()])
        });
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                "oof".to_string(),
                HandleRef {
                    handle: 44,
                    kind: HandleKind::Prediction,
                    owner_controller: self.id.clone(),
                },
            )]),
            predictions: vec![PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: PredictionPartition::Validation,
                fold_id: task.fold_id.clone(),
                sample_ids: prediction_sample_ids.clone(),
                values: vec![vec![1.0]; prediction_sample_ids.len()],
                target_names: vec!["y".to_string()],
            }],
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:{}:{:?}:{}:probe",
                    task.node_plan.node_id,
                    task.phase,
                    task.fold_id
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "nofold".to_string())
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

impl RuntimeController for MockController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        for binding in &task.node_plan.data_bindings {
            let key = format!("data:{}", binding.input_name);
            let handle = task.input_handles.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive data handle `{key}`",
                    task.node_plan.node_id
                ))
            })?;
            if !matches!(handle.kind, HandleKind::Data | HandleKind::DataView) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received non-data/data-view handle for `{key}`",
                    task.node_plan.node_id
                )));
            }
            if !task.data_views.contains_key(&key) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive data view spec for `{key}`",
                    task.node_plan.node_id
                )));
            }
            if task.phase == Phase::FitCv && task.fold_id.is_some() {
                let validation_key = format!("{key}:validation");
                let validation_view = task.data_views.get(&validation_key).ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "node `{}` did not receive validation data view spec for `{validation_key}`",
                            task.node_plan.node_id
                        ))
                    })?;
                if validation_view.partition != DataRequestPartition::FoldValidation {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-validation data view for `{validation_key}`",
                        task.node_plan.node_id
                    )));
                }
            }
        }
        let variant_label = task
            .variant_id
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| "base".to_string());
        let fold_label = task
            .fold_id
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| "nofold".to_string());
        let output = HandleRef {
            handle: self.handle,
            kind: HandleKind::Data,
            owner_controller: self.id.clone(),
        };
        let prediction_output = HandleRef {
            handle: self.handle,
            kind: HandleKind::Prediction,
            owner_controller: self.id.clone(),
        };
        let prediction_sample_ids = validation_view_sample_ids(task)
            .map(|ids| ids.into_iter().collect::<Vec<_>>())
            .unwrap_or_else(|| vec![SampleId::new("s1").unwrap()]);
        let predictions = self
            .emit_prediction
            .then(|| PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: PredictionPartition::Validation,
                fold_id: task.fold_id.clone(),
                sample_ids: prediction_sample_ids.clone(),
                values: vec![vec![1.0]; prediction_sample_ids.len()],
                target_names: vec!["y".to_string()],
            })
            .into_iter()
            .collect::<Vec<_>>();
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([
                ("out".to_string(), output.clone()),
                ("x".to_string(), output.clone()),
                ("x_out".to_string(), output),
                ("pred".to_string(), prediction_output.clone()),
                ("oof".to_string(), prediction_output),
            ]),
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:{}:{:?}:{variant_label}:{fold_label}",
                    task.node_plan.node_id, task.phase
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

struct ReplayMockController {
    id: ControllerId,
    handle: u64,
    require_artifact: bool,
    emit_prediction: bool,
    emit_refit_artifact: bool,
}

impl RuntimeController for ReplayMockController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        for binding in &task.node_plan.data_bindings {
            let key = format!("data:{}", binding.input_name);
            let handle = task.input_handles.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive data handle `{key}`",
                    task.node_plan.node_id
                ))
            })?;
            if !matches!(handle.kind, HandleKind::Data | HandleKind::DataView) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received non-data/data-view handle for `{key}`",
                    task.node_plan.node_id
                )));
            }
            if !task.data_views.contains_key(&key) {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive data view spec for `{key}`",
                    task.node_plan.node_id
                )));
            }
            if task.phase == Phase::FitCv && task.fold_id.is_some() {
                let validation_key = format!("{key}:validation");
                let validation_view = task.data_views.get(&validation_key).ok_or_else(|| {
                        DagMlError::RuntimeValidation(format!(
                            "node `{}` did not receive validation data view spec for `{validation_key}`",
                            task.node_plan.node_id
                        ))
                    })?;
                if validation_view.partition != DataRequestPartition::FoldValidation {
                    return Err(DagMlError::RuntimeValidation(format!(
                        "node `{}` received non-validation data view for `{validation_key}`",
                        task.node_plan.node_id
                    )));
                }
            }
        }
        if self.require_artifact {
            let artifact_id = ArtifactId::new("artifact:model:base:refit").unwrap();
            let key = refit_artifact_input_key(&artifact_id);
            let handle = task.input_handles.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive refit artifact handle `{key}`",
                    task.node_plan.node_id
                ))
            })?;
            if handle.kind != HandleKind::Model {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received non-model refit handle for `{key}`",
                    task.node_plan.node_id
                )));
            }
            let artifact_input = task.artifact_inputs.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "node `{}` did not receive refit artifact metadata `{key}`",
                    task.node_plan.node_id
                ))
            })?;
            if artifact_input.artifact.id != artifact_id
                || artifact_input.node_id != task.node_plan.node_id
                || artifact_input.controller_id != task.node_plan.controller_id
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "node `{}` received mismatched refit artifact metadata `{key}`",
                    task.node_plan.node_id
                )));
            }
        }

        let output = HandleRef {
            handle: self.handle,
            kind: HandleKind::Data,
            owner_controller: self.id.clone(),
        };
        let predictions = self
            .emit_prediction
            .then(|| PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: PredictionPartition::Final,
                fold_id: None,
                sample_ids: vec![SampleId::new("sample:mock").unwrap()],
                values: vec![vec![self.handle as f64]],
                target_names: vec!["y".to_string()],
            })
            .into_iter()
            .collect::<Vec<_>>();
        let artifacts = if self.emit_refit_artifact && task.phase == Phase::Refit {
            vec![ArtifactRef {
                id: ArtifactId::new(format!("artifact:{}:refit", task.node_plan.node_id)).unwrap(),
                kind: "mock_model".to_string(),
                controller_id: self.id.clone(),
                backend: None,
                uri: None,
                content_fingerprint: None,
                size_bytes: Some(128),
                plugin: None,
                plugin_version: None,
            }]
        } else {
            Vec::new()
        };
        let artifact_handles = artifacts
            .iter()
            .map(|artifact| {
                (
                    artifact.id.clone(),
                    HandleRef {
                        handle: self.handle + 10_000,
                        kind: HandleKind::Model,
                        owner_controller: self.id.clone(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([("out".to_string(), output)]),
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: artifacts.clone(),
            artifact_handles,
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:replay:{}:{:?}",
                    task.node_plan.node_id, task.phase
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: artifacts,
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

#[derive(Clone, Copy)]
enum OofSampleMode {
    Aligned,
    Swapped,
}

struct OofEdgeController {
    id: ControllerId,
    base_partition: Option<PredictionPartition>,
    sample_mode: OofSampleMode,
}

impl RuntimeController for OofEdgeController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        if task.node_plan.node_id.as_str() == "model:meta" {
            let handle = task.input_handles.get("model:base.pred").ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "meta node did not receive OOF prediction input".to_string(),
                )
            })?;
            if handle.kind != HandleKind::Prediction {
                return Err(DagMlError::RuntimeValidation(format!(
                    "meta node received {:?} instead of OOF prediction input",
                    handle.kind
                )));
            }
            let prediction_input =
                task.prediction_inputs
                    .get("model:base.pred")
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation(
                            "meta node did not receive OOF prediction input spec".to_string(),
                        )
                    })?;
            if prediction_input.producer_node.as_str() != "model:base"
                || prediction_input.partition != PredictionPartition::Validation
                || prediction_input.prediction_level != PredictionLevel::Sample
                || prediction_input.prediction_width != 1
            {
                return Err(DagMlError::RuntimeValidation(
                    "meta node received invalid OOF prediction input spec".to_string(),
                ));
            }
            if task.phase == Phase::FitCv {
                if prediction_input.fold_id != task.fold_id {
                    return Err(DagMlError::RuntimeValidation(
                        "meta node received OOF prediction spec for the wrong fold".to_string(),
                    ));
                }
                if prediction_input.sample_ids != aligned_validation_samples(task) {
                    return Err(DagMlError::RuntimeValidation(
                        "meta node received OOF prediction spec for wrong samples".to_string(),
                    ));
                }
            }
            if task.phase == Phase::Refit
                && (prediction_input.fold_id.is_some()
                    || prediction_input.fold_ids
                        != vec![
                            FoldId::new("fold:0").unwrap(),
                            FoldId::new("fold:1").unwrap(),
                        ]
                    || prediction_input.sample_ids
                        != vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()])
            {
                return Err(DagMlError::RuntimeValidation(
                    "meta node received invalid refit OOF coverage spec".to_string(),
                ));
            }
        }

        let predictions = if task.node_plan.node_id.as_str() == "model:base" {
            self.base_partition
                .clone()
                .map(|partition| {
                    let sample_ids = match self.sample_mode {
                        OofSampleMode::Aligned => aligned_validation_samples(task),
                        OofSampleMode::Swapped => swapped_validation_samples(task),
                    };
                    let fold_id = matches!(
                        partition,
                        PredictionPartition::Train | PredictionPartition::Validation
                    )
                    .then(|| task.fold_id.clone())
                    .flatten();
                    PredictionBlock {
                        prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                        producer_node: task.node_plan.node_id.clone(),
                        partition,
                        fold_id,
                        sample_ids: sample_ids.clone(),
                        values: vec![vec![0.5]; sample_ids.len()],
                        target_names: vec!["y".to_string()],
                    }
                })
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

        let handle_id = if task.node_plan.node_id.as_str() == "model:base" {
            101
        } else {
            202
        };
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                "pred".to_string(),
                HandleRef {
                    handle: handle_id,
                    kind: HandleKind::Data,
                    owner_controller: self.id.clone(),
                },
            )]),
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:oof:{}:{}",
                    task.node_plan.node_id,
                    task.fold_id
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "nofold".to_string())
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

struct ExpectedRefitOofController {
    id: ControllerId,
    expected_fold_ids: Vec<FoldId>,
    expected_sample_ids: Vec<SampleId>,
    expected_target_names: Vec<String>,
}

impl RuntimeController for ExpectedRefitOofController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        if task.node_plan.node_id.as_str() == "model:meta" && task.phase == Phase::Refit {
            let handle = task.input_handles.get("model:base.pred").ok_or_else(|| {
                DagMlError::RuntimeValidation(
                    "meta node did not receive grouped OOF prediction input".to_string(),
                )
            })?;
            if handle.kind != HandleKind::Prediction {
                return Err(DagMlError::RuntimeValidation(format!(
                    "meta node received {:?} instead of grouped OOF prediction input",
                    handle.kind
                )));
            }
            let prediction_input =
                task.prediction_inputs
                    .get("model:base.pred")
                    .ok_or_else(|| {
                        DagMlError::RuntimeValidation(
                            "meta node did not receive grouped OOF prediction input spec"
                                .to_string(),
                        )
                    })?;
            if prediction_input.producer_node.as_str() != "model:base"
                || prediction_input.source_port != "pred"
                || prediction_input.target_port != "pred"
                || prediction_input.partition != PredictionPartition::Validation
                || prediction_input.prediction_level != PredictionLevel::Sample
                || prediction_input.fold_id.is_some()
                || prediction_input.fold_ids != self.expected_fold_ids
                || prediction_input.sample_ids != self.expected_sample_ids
                || prediction_input.prediction_width != 1
                || prediction_input.target_names != self.expected_target_names
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "meta node received invalid grouped refit OOF spec: {:?}",
                    prediction_input
                )));
            }
        }

        let handle_id = if task.node_plan.node_id.as_str() == "model:base" {
            303
        } else {
            404
        };
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                "pred".to_string(),
                HandleRef {
                    handle: handle_id,
                    kind: HandleKind::Prediction,
                    owner_controller: self.id.clone(),
                },
            )]),
            predictions: Vec::new(),
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:grouped-oof:{}:{:?}",
                    task.node_plan.node_id, task.phase
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: None,
                aggregation_policy_fingerprint: None,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

struct GroupAggregatedOofController {
    id: ControllerId,
}

impl RuntimeController for GroupAggregatedOofController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        if task.node_plan.node_id.as_str() == "model:meta" {
            validate_group_oof_prediction_input(task)?;
        }

        let observation_predictions =
            if task.node_plan.node_id.as_str() == "model:base" && task.phase == Phase::FitCv {
                let (observation_ids, values) =
                    match task.fold_id.as_ref().map(ToString::to_string).as_deref() {
                        Some("fold:0") => (
                            vec![
                                ObservationId::new("obs.S001.base").unwrap(),
                                ObservationId::new("obs.S001.rep1").unwrap(),
                            ],
                            vec![vec![2.0], vec![6.0]],
                        ),
                        Some("fold:1") => (
                            vec![ObservationId::new("obs.S002.base").unwrap()],
                            vec![vec![10.0]],
                        ),
                        _ => (Vec::new(), Vec::new()),
                    };
                if observation_ids.is_empty() {
                    Vec::new()
                } else {
                    vec![ObservationPredictionBlock {
                        prediction_id: Some(format!(
                            "pred:group-oof:{}",
                            task.fold_id
                                .as_ref()
                                .map(ToString::to_string)
                                .unwrap_or_else(|| "nofold".to_string())
                        )),
                        producer_node: task.node_plan.node_id.clone(),
                        partition: PredictionPartition::Validation,
                        fold_id: task.fold_id.clone(),
                        observation_ids,
                        values,
                        weights: Vec::new(),
                        target_names: vec!["y".to_string()],
                    }]
                }
            } else {
                Vec::new()
            };

        let handle_id = if task.node_plan.node_id.as_str() == "model:base" {
            707
        } else {
            808
        };
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                "pred".to_string(),
                HandleRef {
                    handle: handle_id,
                    kind: HandleKind::Prediction,
                    owner_controller: self.id.clone(),
                },
            )]),
            predictions: Vec::new(),
            observation_predictions,
            aggregated_predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:group-oof:{}:{}:{:?}",
                    task.node_plan.node_id,
                    task.fold_id
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "nofold".to_string()),
                    task.phase
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: task
                    .node_plan
                    .shape_plan
                    .as_ref()
                    .map(stable_json_fingerprint)
                    .transpose()?,
                aggregation_policy_fingerprint: task
                    .node_plan
                    .shape_plan
                    .as_ref()
                    .map(|shape_plan| stable_json_fingerprint(&shape_plan.aggregation_policy))
                    .transpose()?,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

fn validate_group_oof_prediction_input(task: &NodeTask) -> Result<()> {
    let handle = task.input_handles.get("model:base.pred").ok_or_else(|| {
        DagMlError::RuntimeValidation(
            "meta node did not receive group OOF prediction input".to_string(),
        )
    })?;
    if handle.kind != HandleKind::Prediction {
        return Err(DagMlError::RuntimeValidation(format!(
            "meta node received {:?} instead of group OOF prediction input",
            handle.kind
        )));
    }
    let prediction_input = task
        .prediction_inputs
        .get("model:base.pred")
        .ok_or_else(|| {
            DagMlError::RuntimeValidation(
                "meta node did not receive group OOF prediction input spec".to_string(),
            )
        })?;
    let fold0 = FoldId::new("fold:0").unwrap();
    let fold1 = FoldId::new("fold:1").unwrap();
    let plant_a = PredictionUnitId::Group(GroupId::new("plant.A").unwrap());
    let plant_b = PredictionUnitId::Group(GroupId::new("plant.B").unwrap());
    let (expected_fold_id, expected_fold_ids, expected_unit_ids) = match task.phase {
        Phase::FitCv => match task.fold_id.as_ref().map(ToString::to_string).as_deref() {
            Some("fold:0") => (Some(fold0.clone()), vec![fold0], vec![plant_a]),
            Some("fold:1") => (Some(fold1.clone()), vec![fold1], vec![plant_b]),
            other => {
                return Err(DagMlError::RuntimeValidation(format!(
                    "unexpected group OOF fold scope {other:?}"
                )));
            }
        },
        Phase::Refit => (None, vec![fold0, fold1], vec![plant_a, plant_b]),
        _ => {
            return Err(DagMlError::RuntimeValidation(format!(
                "unexpected group OOF phase {:?}",
                task.phase
            )));
        }
    };
    if prediction_input.producer_node.as_str() != "model:base"
        || prediction_input.source_port != "pred"
        || prediction_input.target_port != "pred"
        || prediction_input.partition != PredictionPartition::Validation
        || prediction_input.prediction_level != PredictionLevel::Group
        || prediction_input.fold_id != expected_fold_id
        || prediction_input.fold_ids != expected_fold_ids
        || prediction_input.unit_ids != expected_unit_ids
        || !prediction_input.sample_ids.is_empty()
        || prediction_input.prediction_width != 1
        || prediction_input.target_names != vec!["y".to_string()]
    {
        return Err(DagMlError::RuntimeValidation(format!(
            "meta node received invalid group OOF spec: {:?}",
            prediction_input
        )));
    }
    Ok(())
}

struct CustomAggregationController {
    id: ControllerId,
    task_ids: Arc<Mutex<Vec<String>>>,
}

impl RuntimeController for CustomAggregationController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        Err(DagMlError::RuntimeValidation(format!(
            "custom aggregation controller received unexpected node task `{}`",
            task.node_plan.node_id
        )))
    }

    fn invoke_aggregation(
        &self,
        task: &AggregationControllerTask,
    ) -> Result<AggregationControllerResult> {
        self.task_ids.lock().unwrap().push(task.task_id.clone());
        match &task.input {
            AggregationControllerInput::ObservationToSample {
                block,
                relations,
                requested_sample_order,
            } => {
                let mut by_sample = BTreeMap::<SampleId, Vec<Vec<f64>>>::new();
                for (observation_id, row) in block.observation_ids.iter().zip(block.values.iter()) {
                    let sample_id = relations
                        .sample_for_observation(observation_id)
                        .ok_or_else(|| {
                            DagMlError::OofValidation(format!(
                                "missing relation for `{observation_id}`"
                            ))
                        })?;
                    by_sample
                        .entry(sample_id.clone())
                        .or_default()
                        .push(row.clone());
                }
                let values = requested_sample_order
                    .iter()
                    .map(|sample_id| {
                        let rows = by_sample.get(sample_id).ok_or_else(|| {
                            DagMlError::OofValidation(format!(
                                "missing sample `{sample_id}` for custom aggregation"
                            ))
                        })?;
                        let width = rows.first().map_or(0, Vec::len);
                        Ok((0..width)
                            .map(|col| {
                                rows.iter().map(|row| row[col]).sum::<f64>() / rows.len() as f64
                            })
                            .collect::<Vec<_>>())
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(AggregationControllerResult {
                    schema_version:
                        crate::aggregation::AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION,
                    task_id: task.task_id.clone(),
                    output: AggregationControllerOutput::Sample {
                        block: PredictionBlock {
                            prediction_id: block.prediction_id.clone(),
                            producer_node: block.producer_node.clone(),
                            partition: block.partition.clone(),
                            fold_id: block.fold_id.clone(),
                            sample_ids: requested_sample_order.clone(),
                            values,
                            target_names: block.target_names.clone(),
                        },
                    },
                })
            }
            AggregationControllerInput::SampleToUnit {
                block,
                relations,
                requested_unit_order,
            } => {
                let mut by_unit = BTreeMap::<PredictionUnitId, Vec<Vec<f64>>>::new();
                for (sample_id, row) in block.sample_ids.iter().zip(block.values.iter()) {
                    let unit_id = relations
                        .group_for_sample(sample_id)
                        .cloned()
                        .map(PredictionUnitId::Group)
                        .ok_or_else(|| {
                            DagMlError::OofValidation(format!(
                                "missing group relation for `{sample_id}`"
                            ))
                        })?;
                    by_unit.entry(unit_id).or_default().push(row.clone());
                }
                let values = requested_unit_order
                    .iter()
                    .map(|unit_id| {
                        let rows = by_unit.get(unit_id).ok_or_else(|| {
                            DagMlError::OofValidation(format!(
                                "missing unit `{unit_id}` for custom aggregation"
                            ))
                        })?;
                        let width = rows.first().map_or(0, Vec::len);
                        Ok((0..width)
                            .map(|col| rows.iter().map(|row| row[col]).fold(f64::MIN, f64::max))
                            .collect::<Vec<_>>())
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(AggregationControllerResult {
                    schema_version:
                        crate::aggregation::AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION,
                    task_id: task.task_id.clone(),
                    output: AggregationControllerOutput::Unit {
                        block: AggregatedPredictionBlock {
                            prediction_id: block.prediction_id.clone(),
                            producer_node: block.producer_node.clone(),
                            partition: block.partition.clone(),
                            fold_id: block.fold_id.clone(),
                            level: PredictionLevel::Group,
                            unit_ids: requested_unit_order.clone(),
                            values,
                            target_names: block.target_names.clone(),
                        },
                    },
                })
            }
        }
    }
}

struct ObservationPredictionRuntimeController {
    id: ControllerId,
}

impl RuntimeController for ObservationPredictionRuntimeController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        let (observation_ids, values) =
            match task.fold_id.as_ref().map(ToString::to_string).as_deref() {
                Some("fold:0") => (
                    vec![
                        ObservationId::new("obs.S001.base").unwrap(),
                        ObservationId::new("obs.S001.rep1").unwrap(),
                    ],
                    vec![vec![2.0], vec![6.0]],
                ),
                Some("fold:1") => (
                    vec![ObservationId::new("obs.S002.base").unwrap()],
                    vec![vec![10.0]],
                ),
                _ => (
                    vec![ObservationId::new("obs.S001.base").unwrap()],
                    vec![vec![2.0]],
                ),
            };
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([(
                "pred".to_string(),
                HandleRef {
                    handle: 515,
                    kind: HandleKind::Prediction,
                    owner_controller: self.id.clone(),
                },
            )]),
            predictions: Vec::new(),
            observation_predictions: vec![ObservationPredictionBlock {
                prediction_id: Some("pred:obs.runtime".to_string()),
                producer_node: task.node_plan.node_id.clone(),
                partition: PredictionPartition::Validation,
                fold_id: task.fold_id.clone(),
                observation_ids,
                values,
                weights: Vec::new(),
                target_names: vec!["y".to_string()],
            }],
            aggregated_predictions: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:obs-runtime:{}:{}",
                    task.node_plan.node_id,
                    task.fold_id
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "nofold".to_string())
                ))
                .unwrap(),
                run_id: task.run_id.clone(),
                node_id: task.node_plan.node_id.clone(),
                phase: task.phase,
                controller_id: self.id.clone(),
                controller_version: task.node_plan.controller_version.clone(),
                variant_id: task.variant_id.clone(),
                fold_id: task.fold_id.clone(),
                branch_path: task.branch_path.clone(),
                input_lineage: Vec::new(),
                artifact_refs: Vec::new(),
                params_fingerprint: task.node_plan.params_fingerprint.clone(),
                data_model_shape_fingerprint: task
                    .node_plan
                    .shape_plan
                    .as_ref()
                    .map(stable_json_fingerprint)
                    .transpose()?,
                aggregation_policy_fingerprint: task
                    .node_plan
                    .shape_plan
                    .as_ref()
                    .map(|shape_plan| stable_json_fingerprint(&shape_plan.aggregation_policy))
                    .transpose()?,
                seed: task.seed,
                unsafe_flags: BTreeSet::new(),
                metrics: BTreeMap::new(),
            },
        })
    }
}

fn aligned_validation_samples(task: &NodeTask) -> Vec<SampleId> {
    match task.fold_id.as_ref().map(ToString::to_string).as_deref() {
        Some("fold:0") => vec![SampleId::new("s1").unwrap()],
        Some("fold:1") => vec![SampleId::new("s2").unwrap()],
        _ => vec![SampleId::new("s1").unwrap()],
    }
}

fn swapped_validation_samples(task: &NodeTask) -> Vec<SampleId> {
    match task.fold_id.as_ref().map(ToString::to_string).as_deref() {
        Some("fold:0") => vec![SampleId::new("s2").unwrap()],
        Some("fold:1") => vec![SampleId::new("s1").unwrap()],
        _ => vec![SampleId::new("s2").unwrap()],
    }
}

fn temp_prediction_cache_dir(label: &str) -> PathBuf {
    let suffix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after UNIX_EPOCH")
        .as_nanos();
    std::env::temp_dir().join(format!("{label}_{}_{}", std::process::id(), suffix))
}

fn port(name: &str, kind: PortKind) -> PortSpec {
    PortSpec {
        name: name.to_string(),
        kind,
        representation: None,
        cardinality: PortCardinality::One,
        description: String::new(),
    }
}

fn node(id: &str, kind: NodeKind, inputs: Vec<PortSpec>, outputs: Vec<PortSpec>) -> NodeSpec {
    NodeSpec {
        id: NodeId::new(id).unwrap(),
        kind,
        operator: None,
        params: BTreeMap::new(),
        ports: PortSchema { inputs, outputs },
        metadata: BTreeMap::new(),
        seed_label: None,
    }
}

fn controller_manifest(id: &str, kind: NodeKind) -> ControllerManifest {
    let mut capabilities = BTreeSet::from([
        ControllerCapability::Deterministic,
        ControllerCapability::ThreadSafe,
        ControllerCapability::ProcessSafe,
    ]);
    if kind == NodeKind::Model {
        capabilities.insert(ControllerCapability::EmitsPredictions);
        capabilities.insert(ControllerCapability::ConsumesOofPredictions);
        capabilities.insert(ControllerCapability::EmitsArtifacts);
        capabilities.insert(ControllerCapability::Stateful);
    }
    ControllerManifest {
        controller_id: ControllerId::new(id).unwrap(),
        controller_version: "0.1.0".to_string(),
        operator_kind: kind,
        priority: 0,
        supported_phases: BTreeSet::from([Phase::FitCv]),
        input_ports: Vec::new(),
        output_ports: Vec::new(),
        data_requirements: None,
        capabilities,
        operator_selectors: Vec::new(),
        fit_scope: ControllerFitScope::FoldTrain,
        rng_policy: RngPolicy::UsesCoreSeed,
        artifact_policy: ArtifactPolicy::Serializable,
    }
}

fn aggregation_dispatch_plan(with_capability: bool) -> ExecutionPlan {
    let graph = GraphSpec {
        id: "graph:aggregation.dispatch".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![node(
            "aggregate:custom",
            NodeKind::Aggregator,
            Vec::new(),
            Vec::new(),
        )],
        edges: Vec::new(),
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    };
    let campaign = CampaignSpec {
        id: "campaign:aggregation.dispatch".to_string(),
        root_seed: Some(7),
        leakage_policy: Default::default(),
        aggregation_policy: Default::default(),
        split_invocation: None,
        generation: Default::default(),
        shape_plans: BTreeMap::new(),
        data_bindings: BTreeMap::new(),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    };
    let mut manifest = controller_manifest("controller:agg.custom", NodeKind::Aggregator);
    if with_capability {
        manifest
            .capabilities
            .insert(ControllerCapability::AggregatesPredictions);
    }
    let mut registry = ControllerRegistry::new();
    registry.register(manifest).unwrap();
    build_execution_plan("plan:aggregation.dispatch", graph, campaign, &registry).unwrap()
}

fn observation_prediction_runtime_plan() -> ExecutionPlan {
    let model_id = NodeId::new("model:obs").unwrap();
    let graph = GraphSpec {
        id: "graph:observation.prediction.runtime".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![
            node(
                model_id.as_str(),
                NodeKind::Model,
                vec![port("x", PortKind::Data)],
                vec![port("pred", PortKind::Prediction)],
            ),
            node(
                "aggregate:custom",
                NodeKind::Aggregator,
                Vec::new(),
                Vec::new(),
            ),
        ],
        edges: Vec::new(),
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    };
    let mut shape_plans = BTreeMap::new();
    shape_plans.insert(
        model_id.clone(),
        DataModelShapePlan {
            node_id: model_id.clone(),
            input_granularity: Granularity::Observation,
            target_granularity: Granularity::Sample,
            fit_rows: FitBoundary::FoldTrain,
            predict_rows: FitBoundary::FoldValidation,
            feature_namespace: Some("nir".to_string()),
            feature_schema_fingerprint: None,
            target_space: "regression:y".to_string(),
            aggregation_policy: custom_aggregation_policy(PredictionLevel::Sample),
            augmentation_policy: Default::default(),
            selection_policy: Default::default(),
        },
    );
    let mut data_bindings = BTreeMap::new();
    data_bindings.insert(model_id.clone(), vec![data_binding(&model_id)]);
    let campaign = CampaignSpec {
        id: "campaign:observation.prediction.runtime".to_string(),
        root_seed: Some(17),
        leakage_policy: Default::default(),
        aggregation_policy: Default::default(),
        split_invocation: Some(SplitInvocation {
            id: "split:single".to_string(),
            controller_id: None,
            leakage_policy: Default::default(),
            params: BTreeMap::new(),
            fold_set: Some(FoldSet {
                id: "folds:single".to_string(),
                sample_ids: vec![
                    SampleId::new("sample:1").unwrap(),
                    SampleId::new("sample:2").unwrap(),
                ],
                folds: vec![
                    FoldAssignment {
                        fold_id: FoldId::new("fold:0").unwrap(),
                        train_sample_ids: vec![SampleId::new("sample:2").unwrap()],
                        validation_sample_ids: vec![SampleId::new("sample:1").unwrap()],
                        metadata: BTreeMap::new(),
                    },
                    FoldAssignment {
                        fold_id: FoldId::new("fold:1").unwrap(),
                        train_sample_ids: vec![SampleId::new("sample:1").unwrap()],
                        validation_sample_ids: vec![SampleId::new("sample:2").unwrap()],
                        metadata: BTreeMap::new(),
                    },
                ],
                sample_groups: BTreeMap::new(),
            }),
        }),
        generation: Default::default(),
        shape_plans,
        data_bindings,
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    };
    let mut model_manifest = controller_manifest("controller:model.obs", NodeKind::Model);
    model_manifest.supported_phases = BTreeSet::from([Phase::FitCv]);
    let mut agg_manifest = controller_manifest("controller:agg.custom", NodeKind::Aggregator);
    agg_manifest.supported_phases = BTreeSet::from([Phase::Plan]);
    agg_manifest.fit_scope = ControllerFitScope::InferenceOnly;
    agg_manifest
        .capabilities
        .insert(ControllerCapability::AggregatesPredictions);
    let mut registry = ControllerRegistry::new();
    registry.register(model_manifest).unwrap();
    registry.register(agg_manifest).unwrap();
    build_execution_plan(
        "plan:observation.prediction.runtime",
        graph,
        campaign,
        &registry,
    )
    .unwrap()
}

fn live_group_oof_runtime_plan() -> ExecutionPlan {
    let base_id = NodeId::new("model:base").unwrap();
    let graph = GraphSpec {
        id: "graph:live.group.oof".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![
            node(
                "model:base",
                NodeKind::Model,
                vec![port("x", PortKind::Data)],
                vec![port("pred", PortKind::Prediction)],
            ),
            node(
                "model:meta",
                NodeKind::Model,
                vec![port("pred", PortKind::Prediction)],
                vec![port("pred", PortKind::Prediction)],
            ),
        ],
        edges: vec![EdgeSpec {
            source: PortRef {
                node_id: NodeId::new("model:base").unwrap(),
                port_name: "pred".to_string(),
            },
            target: PortRef {
                node_id: NodeId::new("model:meta").unwrap(),
                port_name: "pred".to_string(),
            },
            contract: EdgeContract {
                kind: PortKind::Prediction,
                representation: None,
                requires_oof: true,
                requires_fold_alignment: true,
                propagates_lineage: true,
            },
        }],
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    };
    let aggregation_policy = AggregationPolicy {
        aggregation_level: PredictionLevel::Group,
        method: AggregationMethod::Mean,
        ..AggregationPolicy::default()
    };
    let shape_plan = DataModelShapePlan {
        node_id: base_id.clone(),
        input_granularity: Granularity::Observation,
        target_granularity: Granularity::Sample,
        fit_rows: FitBoundary::FoldTrain,
        predict_rows: FitBoundary::FoldValidation,
        feature_namespace: Some("nir".to_string()),
        feature_schema_fingerprint: None,
        target_space: "regression:y".to_string(),
        aggregation_policy,
        augmentation_policy: Default::default(),
        selection_policy: Default::default(),
    };
    let leakage_policy = LeakageUnitPolicy {
        split_unit: SplitUnit::Group,
        require_group_ids: true,
        ..LeakageUnitPolicy::default()
    };
    let campaign = CampaignSpec {
        id: "campaign:live.group.oof".to_string(),
        root_seed: Some(19),
        leakage_policy: leakage_policy.clone(),
        aggregation_policy: Default::default(),
        split_invocation: Some(SplitInvocation {
            id: "split:live.group.oof".to_string(),
            controller_id: None,
            leakage_policy,
            params: BTreeMap::new(),
            fold_set: Some(FoldSet {
                id: "folds:live.group.oof".to_string(),
                sample_ids: vec![
                    SampleId::new("sample:1").unwrap(),
                    SampleId::new("sample:2").unwrap(),
                ],
                folds: vec![
                    FoldAssignment {
                        fold_id: FoldId::new("fold:0").unwrap(),
                        train_sample_ids: vec![SampleId::new("sample:2").unwrap()],
                        validation_sample_ids: vec![SampleId::new("sample:1").unwrap()],
                        metadata: BTreeMap::new(),
                    },
                    FoldAssignment {
                        fold_id: FoldId::new("fold:1").unwrap(),
                        train_sample_ids: vec![SampleId::new("sample:1").unwrap()],
                        validation_sample_ids: vec![SampleId::new("sample:2").unwrap()],
                        metadata: BTreeMap::new(),
                    },
                ],
                sample_groups: BTreeMap::from([
                    (
                        SampleId::new("sample:1").unwrap(),
                        GroupId::new("plant.A").unwrap(),
                    ),
                    (
                        SampleId::new("sample:2").unwrap(),
                        GroupId::new("plant.B").unwrap(),
                    ),
                ]),
            }),
        }),
        generation: Default::default(),
        shape_plans: BTreeMap::from([(base_id.clone(), shape_plan)]),
        data_bindings: BTreeMap::from([(base_id.clone(), vec![data_binding(&base_id)])]),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    };
    let mut manifest = controller_manifest("controller:model", NodeKind::Model);
    manifest.supported_phases = BTreeSet::from([Phase::FitCv, Phase::Refit]);
    let mut registry = ControllerRegistry::new();
    registry.register(manifest).unwrap();
    build_execution_plan("plan:live.group.oof", graph, campaign, &registry).unwrap()
}

fn custom_aggregation_policy(level: PredictionLevel) -> AggregationPolicy {
    AggregationPolicy {
        aggregation_level: level,
        method: AggregationMethod::CustomController,
        custom_controller: Some(AggregationControllerSpec {
            controller_id: ControllerId::new("controller:agg.custom").unwrap(),
            params: json!({"trim": 0.1}),
        }),
        ..AggregationPolicy::default()
    }
}

fn simple_graph() -> GraphSpec {
    GraphSpec {
        id: "g".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![
            node(
                "transform:snv",
                NodeKind::Transform,
                vec![],
                vec![port("x", PortKind::Data)],
            ),
            node(
                "model:pls",
                NodeKind::Model,
                vec![port("x", PortKind::Data)],
                vec![port("pred", PortKind::Prediction)],
            ),
        ],
        edges: vec![EdgeSpec {
            source: PortRef {
                node_id: NodeId::new("transform:snv").unwrap(),
                port_name: "x".to_string(),
            },
            target: PortRef {
                node_id: NodeId::new("model:pls").unwrap(),
                port_name: "x".to_string(),
            },
            contract: EdgeContract {
                kind: PortKind::Data,
                representation: None,
                requires_oof: false,
                requires_fold_alignment: false,
                propagates_lineage: true,
            },
        }],
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    }
}

fn independent_parallel_graph() -> GraphSpec {
    GraphSpec {
        id: "g:parallel".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![
            node(
                "transform:left",
                NodeKind::Transform,
                vec![],
                vec![port("x", PortKind::Data)],
            ),
            node(
                "transform:right",
                NodeKind::Transform,
                vec![],
                vec![port("x", PortKind::Data)],
            ),
        ],
        edges: Vec::new(),
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    }
}

fn parallel_stress_graph() -> GraphSpec {
    const WIDTH: usize = 6;

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut merge_inputs = Vec::new();
    for idx in 0..WIDTH {
        let transform_id = format!("transform:stress.{idx}");
        let model_id = format!("model:stress.{idx}");
        let merge_port = format!("pred{idx}");
        nodes.push(node(
            &transform_id,
            NodeKind::Transform,
            vec![],
            vec![port("x", PortKind::Data)],
        ));
        nodes.push(node(
            &model_id,
            NodeKind::Model,
            vec![port("x", PortKind::Data)],
            vec![port("pred", PortKind::Prediction)],
        ));
        merge_inputs.push(port(&merge_port, PortKind::Prediction));
        edges.push(EdgeSpec {
            source: PortRef {
                node_id: NodeId::new(transform_id).unwrap(),
                port_name: "x".to_string(),
            },
            target: PortRef {
                node_id: NodeId::new(&model_id).unwrap(),
                port_name: "x".to_string(),
            },
            contract: EdgeContract {
                kind: PortKind::Data,
                representation: None,
                requires_oof: false,
                requires_fold_alignment: false,
                propagates_lineage: true,
            },
        });
        edges.push(EdgeSpec {
            source: PortRef {
                node_id: NodeId::new(model_id).unwrap(),
                port_name: "pred".to_string(),
            },
            target: PortRef {
                node_id: NodeId::new("merge:stress").unwrap(),
                port_name: merge_port,
            },
            contract: EdgeContract {
                kind: PortKind::Prediction,
                representation: None,
                requires_oof: false,
                requires_fold_alignment: true,
                propagates_lineage: true,
            },
        });
    }
    nodes.push(node(
        "merge:stress",
        NodeKind::MixedJoin,
        merge_inputs,
        vec![port("merged", PortKind::Data)],
    ));

    GraphSpec {
        id: "g:parallel.stress".to_string(),
        interface: GraphInterface::default(),
        nodes,
        edges,
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    }
}

fn oof_edge_graph() -> GraphSpec {
    GraphSpec {
        id: "g:oof.edge".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![
            node(
                "model:base",
                NodeKind::Model,
                vec![],
                vec![port("pred", PortKind::Prediction)],
            ),
            node(
                "model:meta",
                NodeKind::Model,
                vec![port("pred", PortKind::Prediction)],
                vec![port("pred", PortKind::Prediction)],
            ),
        ],
        edges: vec![EdgeSpec {
            source: PortRef {
                node_id: NodeId::new("model:base").unwrap(),
                port_name: "pred".to_string(),
            },
            target: PortRef {
                node_id: NodeId::new("model:meta").unwrap(),
                port_name: "pred".to_string(),
            },
            contract: EdgeContract {
                kind: PortKind::Prediction,
                representation: None,
                requires_oof: true,
                requires_fold_alignment: true,
                propagates_lineage: true,
            },
        }],
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    }
}

fn runtime_controllers() -> RuntimeControllerRegistry {
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(MockController {
            id: ControllerId::new("controller:transform").unwrap(),
            handle: 1,
            emit_prediction: false,
        }))
        .unwrap();
    controllers
        .register(Box::new(MockController {
            id: ControllerId::new("controller:model").unwrap(),
            handle: 2,
            emit_prediction: true,
        }))
        .unwrap();
    controllers
}

fn oof_edge_runtime_controllers(
    base_partition: Option<PredictionPartition>,
    sample_mode: OofSampleMode,
) -> RuntimeControllerRegistry {
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(OofEdgeController {
            id: ControllerId::new("controller:model").unwrap(),
            base_partition,
            sample_mode,
        }))
        .unwrap();
    controllers
}

fn expected_refit_oof_runtime_controllers(
    expected_fold_ids: Vec<FoldId>,
    expected_sample_ids: Vec<SampleId>,
    expected_target_names: Vec<String>,
) -> RuntimeControllerRegistry {
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(ExpectedRefitOofController {
            id: ControllerId::new("controller:model").unwrap(),
            expected_fold_ids,
            expected_sample_ids,
            expected_target_names,
        }))
        .unwrap();
    controllers
}

fn replay_runtime_controllers() -> RuntimeControllerRegistry {
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(ReplayMockController {
            id: ControllerId::new("controller:transform.mock").unwrap(),
            handle: 11,
            require_artifact: false,
            emit_prediction: false,
            emit_refit_artifact: false,
        }))
        .unwrap();
    controllers
        .register(Box::new(ReplayMockController {
            id: ControllerId::new("controller:model.mock").unwrap(),
            handle: 22,
            require_artifact: true,
            emit_prediction: true,
            emit_refit_artifact: false,
        }))
        .unwrap();
    controllers
}

fn two_fold_set() -> FoldSet {
    FoldSet {
        id: "outer".to_string(),
        sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
        folds: vec![
            FoldAssignment {
                fold_id: FoldId::new("fold:0").unwrap(),
                train_sample_ids: vec![SampleId::new("s2").unwrap()],
                validation_sample_ids: vec![SampleId::new("s1").unwrap()],
                metadata: BTreeMap::new(),
            },
            FoldAssignment {
                fold_id: FoldId::new("fold:1").unwrap(),
                train_sample_ids: vec![SampleId::new("s1").unwrap()],
                validation_sample_ids: vec![SampleId::new("s2").unwrap()],
                metadata: BTreeMap::new(),
            },
        ],
        sample_groups: BTreeMap::new(),
    }
}

fn three_fold_stress_set() -> FoldSet {
    let samples = (0..6)
        .map(|idx| SampleId::new(format!("s{idx}")).unwrap())
        .collect::<Vec<_>>();
    let folds = (0..3)
        .map(|fold_idx| {
            let validation_sample_ids = samples
                .iter()
                .enumerate()
                .filter_map(|(idx, sample_id)| (idx % 3 == fold_idx).then_some(sample_id.clone()))
                .collect::<Vec<_>>();
            let train_sample_ids = samples
                .iter()
                .filter(|sample_id| !validation_sample_ids.contains(sample_id))
                .cloned()
                .collect::<Vec<_>>();
            FoldAssignment {
                fold_id: FoldId::new(format!("fold:{fold_idx}")).unwrap(),
                train_sample_ids,
                validation_sample_ids,
                metadata: BTreeMap::new(),
            }
        })
        .collect::<Vec<_>>();
    FoldSet {
        id: "outer:stress".to_string(),
        sample_ids: samples,
        folds,
        sample_groups: BTreeMap::new(),
    }
}

fn grouped_repetition_fold_set() -> FoldSet {
    let s1 = SampleId::new("s1").unwrap();
    let s1_rep = SampleId::new("s1_rep").unwrap();
    let s2 = SampleId::new("s2").unwrap();
    let s3 = SampleId::new("s3").unwrap();
    FoldSet {
        id: "outer:grouped-repetition".to_string(),
        sample_ids: vec![s1.clone(), s1_rep.clone(), s2.clone(), s3.clone()],
        folds: vec![
            FoldAssignment {
                fold_id: FoldId::new("fold:0").unwrap(),
                train_sample_ids: vec![s2.clone(), s3.clone()],
                validation_sample_ids: vec![s1.clone(), s1_rep.clone()],
                metadata: BTreeMap::new(),
            },
            FoldAssignment {
                fold_id: FoldId::new("fold:1").unwrap(),
                train_sample_ids: vec![s1.clone(), s1_rep.clone(), s3.clone()],
                validation_sample_ids: vec![s2.clone()],
                metadata: BTreeMap::new(),
            },
            FoldAssignment {
                fold_id: FoldId::new("fold:2").unwrap(),
                train_sample_ids: vec![s1.clone(), s1_rep.clone(), s2.clone()],
                validation_sample_ids: vec![s3.clone()],
                metadata: BTreeMap::new(),
            },
        ],
        sample_groups: BTreeMap::from([
            (s1, GroupId::new("group:product1").unwrap()),
            (s1_rep, GroupId::new("group:product1").unwrap()),
            (s2, GroupId::new("group:product2").unwrap()),
            (s3, GroupId::new("group:product3").unwrap()),
        ]),
    }
}

fn grouped_leakage_policy() -> LeakageUnitPolicy {
    LeakageUnitPolicy {
        split_unit: SplitUnit::Group,
        require_group_ids: true,
        ..LeakageUnitPolicy::default()
    }
}

fn sample_relation(
    observation_id: &str,
    sample_id: &str,
    target_id: &str,
    group_id: &str,
    origin_sample_id: Option<&str>,
    is_augmented: bool,
) -> SampleRelation {
    SampleRelation {
        observation_id: ObservationId::new(observation_id).unwrap(),
        sample_id: SampleId::new(sample_id).unwrap(),
        target_id: Some(TargetId::new(target_id).unwrap()),
        group_id: Some(GroupId::new(group_id).unwrap()),
        origin_sample_id: origin_sample_id.map(|value| SampleId::new(value).unwrap()),
        source_id: Some("nir".to_string()),
        is_augmented,
    }
}

fn grouped_repetition_relations() -> SampleRelationSet {
    SampleRelationSet {
        records: vec![
            sample_relation(
                "obs:s1:a",
                "s1",
                "target:product1",
                "group:product1",
                None,
                false,
            ),
            sample_relation(
                "obs:s1:b",
                "s1",
                "target:product1",
                "group:product1",
                None,
                false,
            ),
            sample_relation(
                "obs:s1:aug0",
                "s1",
                "target:product1",
                "group:product1",
                Some("s1"),
                true,
            ),
            sample_relation(
                "obs:s1rep:a",
                "s1_rep",
                "target:product1",
                "group:product1",
                None,
                false,
            ),
            sample_relation(
                "obs:s2:a",
                "s2",
                "target:product2",
                "group:product2",
                None,
                false,
            ),
            sample_relation(
                "obs:s2:b",
                "s2",
                "target:product2",
                "group:product2",
                None,
                false,
            ),
            sample_relation(
                "obs:s3:a",
                "s3",
                "target:product3",
                "group:product3",
                None,
                false,
            ),
        ],
    }
}

fn grouped_oof_campaign(fold_set: FoldSet) -> CampaignSpec {
    let leakage_policy = grouped_leakage_policy();
    CampaignSpec {
        id: "campaign:oof.grouped-repetition".to_string(),
        root_seed: Some(11),
        leakage_policy: leakage_policy.clone(),
        aggregation_policy: AggregationPolicy::default(),
        split_invocation: Some(SplitInvocation {
            id: "split:outer.grouped-repetition".to_string(),
            controller_id: None,
            leakage_policy,
            params: BTreeMap::new(),
            fold_set: Some(fold_set),
        }),
        generation: Default::default(),
        shape_plans: BTreeMap::new(),
        data_bindings: BTreeMap::new(),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    }
}

fn data_binding(node_id: &NodeId) -> crate::data::DataBinding {
    crate::data::DataBinding {
        node_id: node_id.clone(),
        input_name: "x".to_string(),
        request_id: "nir-to-tabular".to_string(),
        schema_fingerprint: "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b"
            .to_string(),
        plan_fingerprint: "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d"
            .to_string(),
        relation_fingerprint: Some(
            "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10".to_string(),
        ),
        output_representation: "tabular_numeric".to_string(),
        feature_set_id: Some("x".to_string()),
        source_ids: vec!["nir".to_string()],
        require_relations: true,
        view_policy: Default::default(),
        metadata: BTreeMap::new(),
    }
}

fn oof_edge_campaign() -> CampaignSpec {
    CampaignSpec {
        id: "campaign:oof.edge".to_string(),
        root_seed: Some(11),
        leakage_policy: Default::default(),
        aggregation_policy: Default::default(),
        split_invocation: Some(SplitInvocation {
            id: "split:outer".to_string(),
            controller_id: None,
            leakage_policy: Default::default(),
            params: BTreeMap::new(),
            fold_set: Some(two_fold_set()),
        }),
        generation: Default::default(),
        shape_plans: BTreeMap::new(),
        data_bindings: BTreeMap::new(),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    }
}

fn parallel_stress_campaign() -> CampaignSpec {
    CampaignSpec {
        id: "campaign:parallel.stress".to_string(),
        root_seed: Some(31),
        leakage_policy: Default::default(),
        aggregation_policy: Default::default(),
        split_invocation: Some(SplitInvocation {
            id: "split:parallel.stress".to_string(),
            controller_id: None,
            leakage_policy: Default::default(),
            params: BTreeMap::new(),
            fold_set: Some(three_fold_stress_set()),
        }),
        generation: GenerationSpec {
            strategy: GenerationStrategy::Cartesian,
            dimensions: vec![GenerationDimension {
                name: "model_family".to_string(),
                choices: ["linear", "tree", "kernel"]
                    .into_iter()
                    .enumerate()
                    .map(|(rank, label)| GenerationChoice {
                        label: label.to_string(),
                        value: json!(label),
                        param_overrides: (0..6)
                            .map(|idx| crate::generation::GenerationParamOverride {
                                node_id: NodeId::new(format!("model:stress.{idx}")).unwrap(),
                                params: BTreeMap::from([
                                    ("family".to_string(), json!(label)),
                                    ("variant_rank".to_string(), json!(rank)),
                                ]),
                            })
                            .collect(),
                    })
                    .collect(),
            }],
            max_variants: Some(3),
        },
        shape_plans: BTreeMap::new(),
        data_bindings: BTreeMap::new(),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    }
}

fn parallel_stress_manifests() -> crate::controller::ControllerRegistry {
    let mut registry = manifests();
    registry
        .register(controller_manifest(
            "controller:mixed_join",
            NodeKind::MixedJoin,
        ))
        .unwrap();
    registry
}

fn manifests() -> crate::controller::ControllerRegistry {
    let mut manifests = crate::controller::ControllerRegistry::new();
    manifests
        .register(controller_manifest(
            "controller:transform",
            NodeKind::Transform,
        ))
        .unwrap();
    manifests
        .register(controller_manifest("controller:model", NodeKind::Model))
        .unwrap();
    manifests
}

fn oof_edge_manifests(phases: BTreeSet<Phase>) -> crate::controller::ControllerRegistry {
    let mut manifest = controller_manifest("controller:model", NodeKind::Model);
    manifest.supported_phases = phases;
    let mut manifests = crate::controller::ControllerRegistry::new();
    manifests.register(manifest).unwrap();
    manifests
}

fn fixture_plan(plan_id: &str) -> ExecutionPlan {
    let graph: GraphSpec =
        serde_json::from_str(include_str!("../../../../examples/minimal_graph.json")).unwrap();
    let campaign: CampaignSpec = serde_json::from_str(include_str!(
        "../../../../examples/campaign_oof_generation.json"
    ))
    .unwrap();
    let manifests: Vec<ControllerManifest> = serde_json::from_str(include_str!(
        "../../../../examples/controller_manifests.json"
    ))
    .unwrap();
    let mut registry = ControllerRegistry::new();
    for manifest in manifests {
        registry.register(manifest).unwrap();
    }
    build_execution_plan(plan_id, graph, campaign, &registry).unwrap()
}

fn replay_bundle(plan: &ExecutionPlan) -> crate::bundle::ExecutionBundle {
    let model_plan = plan
        .node_plans
        .get(&NodeId::new("model:base").unwrap())
        .unwrap();
    build_execution_bundle(
        crate::ids::BundleId::new("bundle:replay").unwrap(),
        plan,
        Some(plan.variants[0].variant_id.clone()),
        BTreeMap::new(),
        vec![RefitArtifactRecord {
            node_id: model_plan.node_id.clone(),
            controller_id: model_plan.controller_id.clone(),
            artifact: ArtifactRef {
                id: ArtifactId::new("artifact:model:base:refit").unwrap(),
                kind: "mock_model".to_string(),
                controller_id: model_plan.controller_id.clone(),
                backend: None,
                uri: None,
                content_fingerprint: None,
                size_bytes: Some(128),
                plugin: None,
                plugin_version: None,
            },
            params_fingerprint: model_plan.params_fingerprint.clone(),
            data_requirement_keys: vec!["model:base.x".to_string()],
            prediction_requirement_keys: Vec::new(),
        }],
    )
    .unwrap()
}

fn replay_request(bundle: &crate::bundle::ExecutionBundle, phase: Phase) -> ReplayPhaseRequest {
    ReplayPhaseRequest {
        bundle_id: bundle.bundle_id.clone(),
        phase,
        data_envelope_keys: vec!["model:base.x".to_string()],
    }
}

fn replay_envelopes() -> BTreeMap<String, ExternalDataPlanEnvelope> {
    BTreeMap::from([(
        "model:base.x".to_string(),
        serde_json::from_str(include_str!(
            "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
        ))
        .unwrap(),
    )])
}

fn replay_data_provider() -> InMemoryDataProvider {
    InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data.provider").unwrap(),
        replay_envelopes().remove("model:base.x").unwrap(),
    )
    .unwrap()
}

fn replay_artifact_store(bundle: &crate::bundle::ExecutionBundle) -> InMemoryArtifactStore {
    let mut store = InMemoryArtifactStore::new();
    let artifact = &bundle.refit_artifacts[0];
    store
        .register(
            artifact,
            HandleRef {
                handle: 9001,
                kind: HandleKind::Model,
                owner_controller: artifact.controller_id.clone(),
            },
        )
        .unwrap();
    store
}

#[test]
fn sequential_scheduler_invokes_mock_controllers_in_topological_order() {
    let plan = build_execution_plan(
        "plan:fitcv",
        simple_graph(),
        CampaignSpec {
            id: "campaign:fitcv".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let controllers = runtime_controllers();
    let mut ctx = RunContext::new(RunId::new("run:1").unwrap(), Some(11));

    let results = SequentialScheduler
        .execute_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(ctx.lineage.len(), 2);
    assert_eq!(ctx.prediction_store.blocks().len(), 1);
    assert_eq!(results[1].node_id.as_str(), "model:pls");
    let transform_lineage = ctx
        .lineage
        .records()
        .find(|record| record.node_id.as_str() == "transform:snv")
        .expect("transform lineage exists");
    let model_lineage = ctx
        .lineage
        .records()
        .find(|record| record.node_id.as_str() == "model:pls")
        .expect("model lineage exists");
    assert_eq!(
        model_lineage.input_lineage,
        vec![transform_lineage.record_id.clone()]
    );
}

#[test]
fn parallel_scheduler_invokes_independent_level_concurrently() {
    struct ConcurrencyProbeController {
        id: ControllerId,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl RuntimeController for ConcurrencyProbeController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            let mut observed = self.max_active.load(Ordering::SeqCst);
            while active > observed
                && self
                    .max_active
                    .compare_exchange(observed, active, Ordering::SeqCst, Ordering::SeqCst)
                    .is_err()
            {
                observed = self.max_active.load(Ordering::SeqCst);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([(
                    "x".to_string(),
                    HandleRef {
                        handle: task.node_plan.node_id.as_str().len() as u64,
                        kind: HandleKind::Data,
                        owner_controller: self.id.clone(),
                    },
                )]),
                predictions: Vec::new(),
                observation_predictions: Vec::new(),
                aggregated_predictions: Vec::new(),
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:parallel:{}",
                        task.node_plan.node_id
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: Vec::new(),
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    assert!(ParallelScheduler::new(0).is_err());
    let plan = build_execution_plan(
        "plan:parallel",
        independent_parallel_graph(),
        CampaignSpec {
            id: "campaign:parallel".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(ConcurrencyProbeController {
            id: ControllerId::new("controller:transform").unwrap(),
            active: Arc::clone(&active),
            max_active: Arc::clone(&max_active),
        }))
        .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:parallel").unwrap(), Some(11));

    let results = ParallelScheduler::new(2)
        .unwrap()
        .execute_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(ctx.lineage.len(), 2);
    assert!(max_active.load(Ordering::SeqCst) >= 2);
}

#[test]
fn parallel_campaign_scheduler_stress_matches_sequential_across_variants_and_folds() {
    struct StressProbeController {
        id: ControllerId,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        invocations: Arc<Mutex<Vec<String>>>,
        pause: bool,
    }

    impl RuntimeController for StressProbeController {
        fn controller_id(&self) -> &ControllerId {
            &self.id
        }

        fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
            assert_stress_inputs(task)?;
            let task_key = stress_task_key(task);
            self.invocations.lock().unwrap().push(task_key.clone());
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            update_max_active(&self.max_active, active);
            if self.pause {
                std::thread::sleep(std::time::Duration::from_millis(8));
            }
            self.active.fetch_sub(1, Ordering::SeqCst);

            let (output_name, output_kind) = match &task.node_plan.kind {
                NodeKind::Model => ("pred", HandleKind::Prediction),
                NodeKind::MixedJoin => ("merged", HandleKind::Data),
                _ => ("x", HandleKind::Data),
            };
            let prediction_value = (stable_test_handle(&task_key) % 10_000) as f64 / 100.0;
            let predictions = matches!(&task.node_plan.kind, NodeKind::Model)
                .then(|| {
                    let sample_ids = stress_validation_samples(task.fold_id.as_ref());
                    PredictionBlock {
                        prediction_id: Some(format!(
                            "prediction:{}:{}:{}",
                            task.node_plan.node_id,
                            task.variant_id
                                .as_ref()
                                .map(ToString::to_string)
                                .unwrap_or_else(|| "variant:base".to_string()),
                            task.fold_id
                                .as_ref()
                                .map(ToString::to_string)
                                .unwrap_or_else(|| "nofold".to_string())
                        )),
                        producer_node: task.node_plan.node_id.clone(),
                        partition: PredictionPartition::Validation,
                        fold_id: task.fold_id.clone(),
                        values: sample_ids
                            .iter()
                            .enumerate()
                            .map(|(idx, _)| vec![prediction_value + idx as f64])
                            .collect(),
                        sample_ids,
                        target_names: vec!["y".to_string()],
                    }
                })
                .into_iter()
                .collect::<Vec<_>>();
            Ok(NodeResult {
                node_id: task.node_plan.node_id.clone(),
                outputs: BTreeMap::from([(
                    output_name.to_string(),
                    HandleRef {
                        handle: stable_test_handle(&task_key),
                        kind: output_kind,
                        owner_controller: self.id.clone(),
                    },
                )]),
                predictions,
                observation_predictions: Vec::new(),
                aggregated_predictions: Vec::new(),
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                lineage: LineageRecord {
                    record_id: LineageId::new(format!(
                        "lineage:stress:{}:{}:{}",
                        task.node_plan.node_id,
                        task.variant_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "variant:base".to_string()),
                        task.fold_id
                            .as_ref()
                            .map(ToString::to_string)
                            .unwrap_or_else(|| "nofold".to_string())
                    ))
                    .unwrap(),
                    run_id: task.run_id.clone(),
                    node_id: task.node_plan.node_id.clone(),
                    phase: task.phase,
                    controller_id: self.id.clone(),
                    controller_version: task.node_plan.controller_version.clone(),
                    variant_id: task.variant_id.clone(),
                    fold_id: task.fold_id.clone(),
                    branch_path: task.branch_path.clone(),
                    input_lineage: Vec::new(),
                    artifact_refs: Vec::new(),
                    params_fingerprint: task.node_plan.params_fingerprint.clone(),
                    data_model_shape_fingerprint: None,
                    aggregation_policy_fingerprint: None,
                    seed: task.seed,
                    unsafe_flags: BTreeSet::new(),
                    metrics: BTreeMap::new(),
                },
            })
        }
    }

    fn stress_runtime_controllers(
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        invocations: Arc<Mutex<Vec<String>>>,
        pause: bool,
    ) -> RuntimeControllerRegistry {
        let mut controllers = RuntimeControllerRegistry::new();
        for id in [
            "controller:transform",
            "controller:model",
            "controller:mixed_join",
        ] {
            controllers
                .register(Box::new(StressProbeController {
                    id: ControllerId::new(id).unwrap(),
                    active: Arc::clone(&active),
                    max_active: Arc::clone(&max_active),
                    invocations: Arc::clone(&invocations),
                    pause,
                }))
                .unwrap();
        }
        controllers
    }

    fn update_max_active(max_active: &AtomicUsize, active: usize) {
        let mut observed = max_active.load(Ordering::SeqCst);
        while active > observed
            && max_active
                .compare_exchange(observed, active, Ordering::SeqCst, Ordering::SeqCst)
                .is_err()
        {
            observed = max_active.load(Ordering::SeqCst);
        }
    }

    fn stress_task_key(task: &NodeTask) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            task.node_plan.node_id,
            task.variant_id
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "variant:base".to_string()),
            task.fold_id
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_else(|| "nofold".to_string()),
            task.seed
                .map(|seed| seed.to_string())
                .unwrap_or_else(|| "noseed".to_string()),
            task.node_plan.params_fingerprint,
        )
    }

    fn stable_test_handle(label: &str) -> u64 {
        label
            .bytes()
            .fold(14_695_981_039_346_656_037, |hash, byte| {
                (hash ^ byte as u64).wrapping_mul(1_099_511_628_211)
            })
    }

    fn stress_validation_samples(fold_id: Option<&FoldId>) -> Vec<SampleId> {
        match fold_id.map(FoldId::as_str) {
            Some("fold:0") => vec![SampleId::new("s0").unwrap(), SampleId::new("s3").unwrap()],
            Some("fold:1") => vec![SampleId::new("s1").unwrap(), SampleId::new("s4").unwrap()],
            Some("fold:2") => vec![SampleId::new("s2").unwrap(), SampleId::new("s5").unwrap()],
            _ => vec![SampleId::new("s0").unwrap()],
        }
    }

    fn assert_stress_inputs(task: &NodeTask) -> Result<()> {
        let node_id = task.node_plan.node_id.as_str();
        if node_id.starts_with("transform:stress.") && !task.input_handles.is_empty() {
            return Err(DagMlError::RuntimeValidation(format!(
                "source node `{node_id}` received unexpected inputs"
            )));
        }
        if node_id.starts_with("model:stress.")
            && !task
                .input_handles
                .keys()
                .any(|key| key.starts_with("transform:stress.") && key.ends_with(".x"))
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "model node `{node_id}` did not receive its transform input"
            )));
        }
        if node_id == "merge:stress" {
            let model_inputs = task
                .input_handles
                .keys()
                .filter(|key| key.starts_with("model:stress.") && key.ends_with(".pred"))
                .count();
            if model_inputs != 6 {
                return Err(DagMlError::RuntimeValidation(format!(
                    "merge node received {model_inputs} model inputs, expected 6"
                )));
            }
        }
        Ok(())
    }

    fn lineage_records(ctx: &RunContext) -> Vec<LineageRecord> {
        ctx.lineage.records().cloned().collect::<Vec<_>>()
    }

    let plan = build_execution_plan(
        "plan:parallel.stress",
        parallel_stress_graph(),
        parallel_stress_campaign(),
        &parallel_stress_manifests(),
    )
    .unwrap();
    let levels = plan.node_parallel_levels_for_phase(Phase::FitCv).unwrap();
    assert_eq!(
        levels.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![6, 6, 1]
    );
    assert_eq!(plan.variants.len(), 3);
    assert_eq!(plan.fold_set.as_ref().unwrap().folds.len(), 3);

    let sequential_active = Arc::new(AtomicUsize::new(0));
    let sequential_max_active = Arc::new(AtomicUsize::new(0));
    let sequential_invocations = Arc::new(Mutex::new(Vec::new()));
    let sequential_controllers = stress_runtime_controllers(
        Arc::clone(&sequential_active),
        Arc::clone(&sequential_max_active),
        Arc::clone(&sequential_invocations),
        false,
    );
    let mut sequential_ctx = RunContext::new(RunId::new("run:parallel.stress").unwrap(), Some(31));
    let sequential_results = SequentialScheduler
        .execute_campaign_phase(
            &plan,
            &sequential_controllers,
            &mut sequential_ctx,
            Phase::FitCv,
        )
        .unwrap();

    let parallel_active = Arc::new(AtomicUsize::new(0));
    let parallel_max_active = Arc::new(AtomicUsize::new(0));
    let parallel_invocations = Arc::new(Mutex::new(Vec::new()));
    let parallel_controllers = stress_runtime_controllers(
        Arc::clone(&parallel_active),
        Arc::clone(&parallel_max_active),
        Arc::clone(&parallel_invocations),
        true,
    );
    let mut parallel_ctx = RunContext::new(RunId::new("run:parallel.stress").unwrap(), Some(31));
    let parallel_results = ParallelScheduler::new(4)
        .unwrap()
        .execute_campaign_phase(
            &plan,
            &parallel_controllers,
            &mut parallel_ctx,
            Phase::FitCv,
        )
        .unwrap();

    assert_eq!(sequential_results.len(), 117);
    assert_eq!(parallel_results, sequential_results);
    assert_eq!(
        parallel_ctx.prediction_store.blocks(),
        sequential_ctx.prediction_store.blocks()
    );
    assert_eq!(
        lineage_records(&parallel_ctx),
        lineage_records(&sequential_ctx)
    );
    assert_eq!(parallel_ctx.prediction_store.blocks().len(), 54);
    assert_eq!(parallel_ctx.lineage.len(), 117);
    assert_eq!(
        parallel_results
            .iter()
            .filter_map(|result| result.lineage.seed)
            .collect::<BTreeSet<_>>()
            .len(),
        parallel_results.len()
    );
    assert_eq!(
        parallel_invocations
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>(),
        sequential_invocations
            .lock()
            .unwrap()
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
    );
    let observed_parallelism = parallel_max_active.load(Ordering::SeqCst);
    assert!((2..=4).contains(&observed_parallelism));
    assert_eq!(parallel_active.load(Ordering::SeqCst), 0);
    assert_eq!(sequential_max_active.load(Ordering::SeqCst), 1);
}

#[test]
fn campaign_scheduler_expands_variants_and_cv_folds() {
    let plan = build_execution_plan(
        "plan:campaign",
        simple_graph(),
        CampaignSpec {
            id: "campaign:fitcv".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: Some(SplitInvocation {
                id: "split:outer".to_string(),
                controller_id: None,
                leakage_policy: Default::default(),
                params: BTreeMap::new(),
                fold_set: Some(two_fold_set()),
            }),
            generation: GenerationSpec {
                strategy: GenerationStrategy::Cartesian,
                dimensions: vec![GenerationDimension {
                    name: "model_family".to_string(),
                    choices: vec![
                        GenerationChoice {
                            label: "pls".to_string(),
                            value: json!("pls"),
                            param_overrides: Vec::new(),
                        },
                        GenerationChoice {
                            label: "rf".to_string(),
                            value: json!("rf"),
                            param_overrides: Vec::new(),
                        },
                    ],
                }],
                max_variants: Some(2),
            },
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let controllers = runtime_controllers();
    let mut ctx = RunContext::new(RunId::new("run:campaign").unwrap(), Some(11));

    let results = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .unwrap();

    assert_eq!(results.len(), 8);
    assert_eq!(ctx.lineage.len(), 8);
    assert_eq!(ctx.prediction_store.blocks().len(), 4);
    assert!(ctx
        .lineage
        .records()
        .all(|record| record.variant_id.is_some() && record.fold_id.is_some()));
    assert_eq!(
        ctx.lineage
            .records()
            .filter_map(|record| record.seed)
            .collect::<BTreeSet<_>>()
            .len(),
        8
    );
}

#[test]
fn node_tasks_expose_generation_variant_context() {
    let plan = build_execution_plan(
        "plan:generation.task.context",
        simple_graph(),
        CampaignSpec {
            id: "campaign:generation.task.context".to_string(),
            root_seed: Some(23),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: None,
            generation: GenerationSpec {
                strategy: GenerationStrategy::Cartesian,
                dimensions: vec![GenerationDimension {
                    name: "model_family".to_string(),
                    choices: vec![
                        GenerationChoice {
                            label: "pls".to_string(),
                            value: json!("pls"),
                            param_overrides: vec![crate::generation::GenerationParamOverride {
                                node_id: NodeId::new("model:pls").unwrap(),
                                params: BTreeMap::from([("n_components".to_string(), json!(4))]),
                            }],
                        },
                        GenerationChoice {
                            label: "rf".to_string(),
                            value: json!("rf"),
                            param_overrides: vec![crate::generation::GenerationParamOverride {
                                node_id: NodeId::new("model:pls").unwrap(),
                                params: BTreeMap::from([("trees".to_string(), json!(64))]),
                            }],
                        },
                    ],
                }],
                max_variants: Some(2),
            },
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let observed_variants = Arc::new(Mutex::new(Vec::new()));
    let observed_node_plans = Arc::new(Mutex::new(Vec::new()));
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(MockController {
            id: ControllerId::new("controller:transform").unwrap(),
            handle: 1,
            emit_prediction: false,
        }))
        .unwrap();
    controllers
        .register(Box::new(VariantProbeController {
            id: ControllerId::new("controller:model").unwrap(),
            handle: 2,
            variants: Arc::clone(&observed_variants),
            node_plans: Arc::clone(&observed_node_plans),
        }))
        .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:generation.task.context").unwrap(), Some(23));

    let results = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .unwrap();

    assert_eq!(results.len(), 4);
    let observed = observed_variants.lock().unwrap();
    assert_eq!(observed.len(), 2);
    let mut labels = BTreeSet::new();
    for variant in observed.iter().map(|variant| variant.as_ref().unwrap()) {
        variant.validate().unwrap();
        let expected = plan
            .variants
            .iter()
            .find(|planned| planned.variant_id == variant.variant_id)
            .unwrap();
        assert_eq!(variant.choices, expected.choices);
        assert_eq!(variant.fingerprint, expected.fingerprint);
        assert_eq!(variant.seed, expected.seed);
        labels.insert(variant.choices["model_family"].label.as_str());
    }
    assert_eq!(labels, BTreeSet::from(["pls", "rf"]));
    let observed_plans = observed_node_plans.lock().unwrap();
    assert_eq!(observed_plans.len(), 2);
    let base_plan = plan
        .node_plans
        .get(&NodeId::new("model:pls").unwrap())
        .unwrap();
    assert!(observed_plans
        .iter()
        .all(|node_plan| node_plan.params_fingerprint != base_plan.params_fingerprint));
    assert!(observed_plans
        .iter()
        .any(|node_plan| node_plan.params.get("n_components") == Some(&json!(4))));
    assert!(observed_plans
        .iter()
        .any(|node_plan| node_plan.params.get("trees") == Some(&json!(64))));
}

#[test]
fn requires_oof_prediction_edge_supplies_validated_prediction_handle() {
    let plan = build_execution_plan(
        "plan:oof.edge.success",
        oof_edge_graph(),
        oof_edge_campaign(),
        &manifests(),
    )
    .unwrap();
    let controllers = oof_edge_runtime_controllers(
        Some(PredictionPartition::Validation),
        OofSampleMode::Aligned,
    );
    let mut ctx = RunContext::new(RunId::new("run:oof.edge.success").unwrap(), Some(11));

    let results = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .unwrap();

    assert_eq!(results.len(), 4);
    assert_eq!(ctx.prediction_store.blocks().len(), 2);
    assert_eq!(
        results
            .iter()
            .filter(|result| result.node_id.as_str() == "model:meta")
            .count(),
        2
    );
}

#[test]
fn requires_oof_prediction_edge_rejects_missing_validation_predictions() {
    let plan = build_execution_plan(
        "plan:oof.edge.missing",
        oof_edge_graph(),
        oof_edge_campaign(),
        &manifests(),
    )
    .unwrap();
    let controllers = oof_edge_runtime_controllers(None, OofSampleMode::Aligned);
    let mut ctx = RunContext::new(RunId::new("run:oof.edge.missing").unwrap(), Some(11));

    let error = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .unwrap_err()
        .to_string();

    assert!(error.contains("requires OOF validation predictions"));
    assert!(error.contains("model:base"));
}

#[test]
fn requires_oof_prediction_edge_rejects_train_predictions_as_features() {
    let plan = build_execution_plan(
        "plan:oof.edge.train",
        oof_edge_graph(),
        oof_edge_campaign(),
        &manifests(),
    )
    .unwrap();
    let controllers =
        oof_edge_runtime_controllers(Some(PredictionPartition::Train), OofSampleMode::Aligned);
    let mut ctx = RunContext::new(RunId::new("run:oof.edge.train").unwrap(), Some(11));

    let error = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .unwrap_err()
        .to_string();

    assert!(error.contains("requires OOF validation predictions"));
}

#[test]
fn requires_oof_prediction_edge_rejects_fold_misalignment() {
    let plan = build_execution_plan(
        "plan:oof.edge.misaligned",
        oof_edge_graph(),
        oof_edge_campaign(),
        &manifests(),
    )
    .unwrap();
    let controllers = oof_edge_runtime_controllers(
        Some(PredictionPartition::Validation),
        OofSampleMode::Swapped,
    );
    let mut ctx = RunContext::new(RunId::new("run:oof.edge.misaligned").unwrap(), Some(11));

    let error = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .unwrap_err()
        .to_string();

    assert!(error.contains("do not match validation samples"));
}

#[test]
fn requires_oof_prediction_edge_refit_uses_cv_oof_coverage() {
    let plan = build_execution_plan(
        "plan:oof.edge.refit",
        oof_edge_graph(),
        oof_edge_campaign(),
        &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
    )
    .unwrap();
    let fit_controllers = oof_edge_runtime_controllers(
        Some(PredictionPartition::Validation),
        OofSampleMode::Aligned,
    );
    let mut ctx = RunContext::new(RunId::new("run:oof.edge.refit").unwrap(), Some(11));
    SequentialScheduler
        .execute_campaign_phase(&plan, &fit_controllers, &mut ctx, Phase::FitCv)
        .unwrap();
    assert_eq!(ctx.prediction_store.blocks().len(), 2);

    let refit_controllers = oof_edge_runtime_controllers(None, OofSampleMode::Aligned);
    let refit_results = SequentialScheduler
        .execute_campaign_phase(&plan, &refit_controllers, &mut ctx, Phase::Refit)
        .unwrap();

    assert_eq!(refit_results.len(), 2);
    assert_eq!(
        refit_results
            .iter()
            .filter(|result| result.node_id.as_str() == "model:meta")
            .count(),
        1
    );
}

#[test]
fn requires_oof_prediction_edge_feeds_live_group_units_to_fit_cv_and_refit() {
    let plan = live_group_oof_runtime_plan();
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(GroupAggregatedOofController {
            id: ControllerId::new("controller:model").unwrap(),
        }))
        .unwrap();
    let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
    ))
    .unwrap();
    let data_provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data").unwrap(),
        envelope,
    )
    .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:live.group.oof").unwrap(), Some(19));

    let fit_results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &data_provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap();

    assert_eq!(fit_results.len(), 4);
    assert!(ctx.prediction_store.blocks().is_empty());
    assert_eq!(ctx.aggregated_prediction_store.blocks().len(), 2);
    assert_eq!(
        ctx.aggregated_prediction_store
            .blocks()
            .iter()
            .map(|block| (&block.fold_id, block.level, block.unit_ids.clone()))
            .collect::<Vec<_>>(),
        vec![
            (
                &Some(FoldId::new("fold:0").unwrap()),
                PredictionLevel::Group,
                vec![PredictionUnitId::Group(GroupId::new("plant.A").unwrap())],
            ),
            (
                &Some(FoldId::new("fold:1").unwrap()),
                PredictionLevel::Group,
                vec![PredictionUnitId::Group(GroupId::new("plant.B").unwrap())],
            ),
        ]
    );

    let refit_results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &data_provider,
            &mut ctx,
            Phase::Refit,
        )
        .unwrap();

    assert_eq!(refit_results.len(), 2);
    assert_eq!(
        refit_results
            .iter()
            .filter(|result| result.node_id.as_str() == "model:meta")
            .count(),
        1
    );
}

#[test]
fn aggregated_oof_edge_rejects_relation_level_train_validation_overlap() {
    let plan = live_group_oof_runtime_plan();
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(GroupAggregatedOofController {
            id: ControllerId::new("controller:model").unwrap(),
        }))
        .unwrap();
    let mut envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
    ))
    .unwrap();
    for record in &mut envelope
        .coordinator_relations
        .as_mut()
        .expect("fixture carries coordinator relations")
        .records
    {
        if record.sample_id.as_str() == "sample:2" {
            record.group_id = Some(GroupId::new("plant.A").unwrap());
        }
    }
    let data_provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data").unwrap(),
        envelope,
    )
    .unwrap();
    let mut ctx = RunContext::new(
        RunId::new("run:live.group.oof.relation-overlap").unwrap(),
        Some(19),
    );

    let error = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &data_provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap_err()
        .to_string();

    assert!(
        error.contains("both train and validation partitions"),
        "unexpected overlap error: {error}"
    );
}

#[test]
fn runtime_dispatches_custom_observation_aggregation_controller() {
    let plan = aggregation_dispatch_plan(true);
    let task_ids = Arc::new(Mutex::new(Vec::new()));
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(CustomAggregationController {
            id: ControllerId::new("controller:agg.custom").unwrap(),
            task_ids: Arc::clone(&task_ids),
        }))
        .unwrap();
    let relations = SampleRelationSet {
        records: vec![
            sample_relation("obs:s1:a", "s1", "target:s1", "group:left", None, false),
            sample_relation("obs:s1:b", "s1", "target:s1", "group:left", None, false),
            sample_relation("obs:s2:a", "s2", "target:s2", "group:right", None, false),
        ],
    };

    let block = dispatch_custom_observation_aggregation(
        &plan,
        &controllers,
        "agg-task:obs-to-sample",
        ObservationPredictionBlock {
            prediction_id: Some("pred:obs".to_string()),
            producer_node: NodeId::new("model:base").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            observation_ids: vec![
                ObservationId::new("obs:s1:a").unwrap(),
                ObservationId::new("obs:s1:b").unwrap(),
                ObservationId::new("obs:s2:a").unwrap(),
            ],
            values: vec![vec![1.0], vec![5.0], vec![10.0]],
            weights: Vec::new(),
            target_names: vec!["y".to_string()],
        },
        relations,
        custom_aggregation_policy(PredictionLevel::Sample),
        vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
    )
    .unwrap();

    assert_eq!(
        block.sample_ids,
        vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()]
    );
    assert_eq!(block.values, vec![vec![3.0], vec![10.0]]);
    assert_eq!(
        task_ids.lock().unwrap().as_slice(),
        &["agg-task:obs-to-sample".to_string()]
    );
}

#[test]
fn runtime_dispatches_custom_sample_to_group_aggregation_controller() {
    let plan = aggregation_dispatch_plan(true);
    let task_ids = Arc::new(Mutex::new(Vec::new()));
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(CustomAggregationController {
            id: ControllerId::new("controller:agg.custom").unwrap(),
            task_ids: Arc::clone(&task_ids),
        }))
        .unwrap();
    let relations = SampleRelationSet {
        records: vec![
            sample_relation("obs:s1:a", "s1", "target:s1", "group:left", None, false),
            sample_relation("obs:s2:a", "s2", "target:s2", "group:left", None, false),
            sample_relation("obs:s3:a", "s3", "target:s3", "group:right", None, false),
        ],
    };
    let left = PredictionUnitId::Group(GroupId::new("group:left").unwrap());
    let right = PredictionUnitId::Group(GroupId::new("group:right").unwrap());

    let block = dispatch_custom_sample_aggregation(
        &plan,
        &controllers,
        "agg-task:sample-to-group",
        PredictionBlock {
            prediction_id: Some("pred:sample".to_string()),
            producer_node: NodeId::new("model:base").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            sample_ids: vec![
                SampleId::new("s1").unwrap(),
                SampleId::new("s2").unwrap(),
                SampleId::new("s3").unwrap(),
            ],
            values: vec![vec![1.0], vec![8.0], vec![3.0]],
            target_names: vec!["y".to_string()],
        },
        relations,
        custom_aggregation_policy(PredictionLevel::Group),
        vec![left.clone(), right.clone()],
    )
    .unwrap();

    assert_eq!(block.level, PredictionLevel::Group);
    assert_eq!(block.unit_ids, vec![left, right]);
    assert_eq!(block.values, vec![vec![8.0], vec![3.0]]);
    assert_eq!(
        task_ids.lock().unwrap().as_slice(),
        &["agg-task:sample-to-group".to_string()]
    );
}

#[test]
fn custom_aggregation_dispatch_requires_controller_capability() {
    let plan = aggregation_dispatch_plan(false);
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(CustomAggregationController {
            id: ControllerId::new("controller:agg.custom").unwrap(),
            task_ids: Arc::new(Mutex::new(Vec::new())),
        }))
        .unwrap();
    let error = dispatch_custom_observation_aggregation(
        &plan,
        &controllers,
        "agg-task:no-capability",
        ObservationPredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:base").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            observation_ids: vec![ObservationId::new("obs:s1:a").unwrap()],
            values: vec![vec![1.0]],
            weights: Vec::new(),
            target_names: vec!["y".to_string()],
        },
        SampleRelationSet {
            records: vec![sample_relation(
                "obs:s1:a",
                "s1",
                "target:s1",
                "group:left",
                None,
                false,
            )],
        },
        custom_aggregation_policy(PredictionLevel::Sample),
        vec![SampleId::new("s1").unwrap()],
    )
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("aggregates_predictions"),
        "unexpected capability error: {error}"
    );
}

#[test]
fn scheduler_aggregates_observation_predictions_with_custom_controller() {
    let plan = observation_prediction_runtime_plan();
    let task_ids = Arc::new(Mutex::new(Vec::new()));
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(ObservationPredictionRuntimeController {
            id: ControllerId::new("controller:model.obs").unwrap(),
        }))
        .unwrap();
    controllers
        .register(Box::new(CustomAggregationController {
            id: ControllerId::new("controller:agg.custom").unwrap(),
            task_ids: Arc::clone(&task_ids),
        }))
        .unwrap();
    let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
    ))
    .unwrap();
    let data_provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data").unwrap(),
        envelope,
    )
    .unwrap();
    let mut ctx = RunContext::new(
        RunId::new("run:observation.prediction.runtime").unwrap(),
        Some(17),
    );

    let results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &data_provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap();

    assert_eq!(results.len(), 2);
    assert!(results
        .iter()
        .all(|result| result.observation_predictions.len() == 1));
    assert!(results.iter().all(|result| result.predictions.len() == 1));
    let blocks = ctx.prediction_store.blocks();
    assert_eq!(
        blocks
            .iter()
            .flat_map(|block| block.sample_ids.iter().cloned())
            .collect::<Vec<_>>(),
        vec![
            SampleId::new("sample:1").unwrap(),
            SampleId::new("sample:2").unwrap()
        ]
    );
    assert_eq!(
        blocks
            .iter()
            .flat_map(|block| block.values.iter().cloned())
            .collect::<Vec<_>>(),
        vec![vec![4.0], vec![10.0]]
    );
    assert_eq!(task_ids.lock().unwrap().len(), 2);
}

#[test]
fn refit_oof_accepts_grouped_repeated_aggregation_and_refuses_origin_leakage() {
    let fold_set = grouped_repetition_fold_set();
    let relations = grouped_repetition_relations();
    let leakage_policy = grouped_leakage_policy();
    relations
        .validate_against_fold_set(&fold_set, &leakage_policy)
        .unwrap();

    let mut leaky_relations = relations.clone();
    leaky_relations.records.push(sample_relation(
        "obs:s1:leaky_aug",
        "s1",
        "target:product1",
        "group:product1",
        Some("s2"),
        true,
    ));
    let leak_error = leaky_relations
        .validate_against_fold_set(&fold_set, &leakage_policy)
        .unwrap_err()
        .to_string();
    assert!(
        leak_error.contains("leaks origin sample"),
        "unexpected leakage error: {leak_error}"
    );

    let plan = build_execution_plan(
        "plan:oof.edge.grouped-repetition.refit",
        oof_edge_graph(),
        grouped_oof_campaign(fold_set.clone()),
        &oof_edge_manifests(BTreeSet::from([Phase::Refit])),
    )
    .unwrap();
    let mut ctx = RunContext::new(
        RunId::new("run:oof.edge.grouped-repetition.refit").unwrap(),
        Some(11),
    );

    let fold0 = aggregate_observation_predictions(
        &ObservationPredictionBlock {
            prediction_id: Some("pred:model:base:fold0:obs".to_string()),
            producer_node: NodeId::new("model:base").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            observation_ids: vec![
                ObservationId::new("obs:s1:a").unwrap(),
                ObservationId::new("obs:s1:b").unwrap(),
                ObservationId::new("obs:s1rep:a").unwrap(),
            ],
            values: vec![vec![1.0], vec![3.0], vec![4.0]],
            weights: Vec::new(),
            target_names: vec!["y".to_string()],
        },
        &relations,
        &AggregationPolicy::default(),
        &[
            SampleId::new("s1").unwrap(),
            SampleId::new("s1_rep").unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(fold0.values, vec![vec![2.0], vec![4.0]]);
    ctx.prediction_store.append(fold0).unwrap();

    let fold1 = aggregate_observation_predictions(
        &ObservationPredictionBlock {
            prediction_id: Some("pred:model:base:fold1:obs".to_string()),
            producer_node: NodeId::new("model:base").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:1").unwrap()),
            observation_ids: vec![
                ObservationId::new("obs:s2:a").unwrap(),
                ObservationId::new("obs:s2:b").unwrap(),
            ],
            values: vec![vec![10.0], vec![14.0]],
            weights: Vec::new(),
            target_names: vec!["y".to_string()],
        },
        &relations,
        &AggregationPolicy::default(),
        &[SampleId::new("s2").unwrap()],
    )
    .unwrap();
    assert_eq!(fold1.values, vec![vec![12.0]]);
    ctx.prediction_store.append(fold1).unwrap();

    let fold2 = aggregate_observation_predictions(
        &ObservationPredictionBlock {
            prediction_id: Some("pred:model:base:fold2:obs".to_string()),
            producer_node: NodeId::new("model:base").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:2").unwrap()),
            observation_ids: vec![ObservationId::new("obs:s3:a").unwrap()],
            values: vec![vec![20.0]],
            weights: Vec::new(),
            target_names: vec!["y".to_string()],
        },
        &relations,
        &AggregationPolicy::default(),
        &[SampleId::new("s3").unwrap()],
    )
    .unwrap();
    assert_eq!(fold2.values, vec![vec![20.0]]);
    ctx.prediction_store.append(fold2).unwrap();
    assert_eq!(ctx.prediction_store.blocks().len(), 3);

    let controllers = expected_refit_oof_runtime_controllers(
        vec![
            FoldId::new("fold:0").unwrap(),
            FoldId::new("fold:1").unwrap(),
            FoldId::new("fold:2").unwrap(),
        ],
        vec![
            SampleId::new("s1").unwrap(),
            SampleId::new("s1_rep").unwrap(),
            SampleId::new("s2").unwrap(),
            SampleId::new("s3").unwrap(),
        ],
        vec!["y".to_string()],
    );
    let refit_results = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::Refit)
        .unwrap();

    assert_eq!(refit_results.len(), 2);
    assert_eq!(
        refit_results
            .iter()
            .filter(|result| result.node_id.as_str() == "model:meta")
            .count(),
        1
    );
}

#[test]
fn in_memory_prediction_cache_store_loads_and_materializes_oof_payloads() {
    let plan = build_execution_plan(
        "plan:oof.edge.cache.store",
        oof_edge_graph(),
        oof_edge_campaign(),
        &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
    )
    .unwrap();
    let fit_controllers = oof_edge_runtime_controllers(
        Some(PredictionPartition::Validation),
        OofSampleMode::Aligned,
    );
    let mut ctx = RunContext::new(RunId::new("run:oof.edge.cache.store").unwrap(), Some(11));
    SequentialScheduler
        .execute_campaign_phase(&plan, &fit_controllers, &mut ctx, Phase::FitCv)
        .unwrap();

    let requirement = BundlePredictionRequirement {
        producer_node: NodeId::new("model:base").unwrap(),
        source_port: "pred".to_string(),
        consumer_node: NodeId::new("model:meta").unwrap(),
        target_port: "pred".to_string(),
        partition: PredictionPartition::Validation,
        prediction_level: PredictionLevel::Sample,
        fold_ids: vec![
            FoldId::new("fold:0").unwrap(),
            FoldId::new("fold:1").unwrap(),
        ],
        unit_ids: Vec::new(),
        sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
        prediction_width: 1,
        target_names: vec!["y".to_string()],
    };
    let cache = build_prediction_cache_record(&requirement, ctx.prediction_store.blocks()).unwrap();
    let payload =
        build_prediction_cache_payload(&requirement, ctx.prediction_store.blocks()).unwrap();
    let bundle = build_execution_bundle_with_prediction_contracts(
        BundleId::new("bundle:oof.edge.cache.store").unwrap(),
        &plan,
        Some(plan.variants[0].variant_id.clone()),
        BTreeMap::new(),
        Vec::new(),
        vec![requirement.clone()],
        vec![cache.clone()],
    )
    .unwrap();
    let payload_set = BundlePredictionCachePayloadSet {
        bundle_id: bundle.bundle_id.clone(),
        schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
        caches: vec![payload],
    };
    let store = InMemoryPredictionCacheStore::from_payloads(&bundle, payload_set).unwrap();
    assert_eq!(store.payload_count(), 1);
    assert_eq!(store.load_blocks(&requirement.key()).unwrap().len(), 2);

    ReplayPhaseRequest {
        bundle_id: bundle.bundle_id.clone(),
        phase: Phase::Refit,
        data_envelope_keys: Vec::new(),
    }
    .validate_for_bundle_with_prediction_cache_store(&bundle, true)
    .unwrap();

    let handle = store
        .materialize(&PredictionCacheMaterializationRequest {
            run_id: RunId::new("run:oof.edge.cache.store.replay").unwrap(),
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Refit,
            variant_id: bundle.selected_variant_id.clone(),
            requirement: requirement.clone(),
            cache,
            producer_controller_id: ControllerId::new("controller:model").unwrap(),
        })
        .unwrap();
    assert_eq!(handle.kind, HandleKind::Prediction);
    assert_eq!(
        handle.owner_controller,
        ControllerId::new("controller:model").unwrap()
    );
    let records = store.materialization_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].requirement_key, requirement.key());
    assert_eq!(records[0].handle, handle);
}

#[test]
fn prediction_cache_stores_load_and_materialize_aggregated_payloads() {
    let plan = build_execution_plan(
        "plan:oof.edge.aggregated.cache.store",
        oof_edge_graph(),
        oof_edge_campaign(),
        &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
    )
    .unwrap();
    let target_a = PredictionUnitId::Target(TargetId::new("target:a").unwrap());
    let target_b = PredictionUnitId::Target(TargetId::new("target:b").unwrap());
    let requirement = BundlePredictionRequirement {
        producer_node: NodeId::new("model:base").unwrap(),
        source_port: "pred".to_string(),
        consumer_node: NodeId::new("model:meta").unwrap(),
        target_port: "pred".to_string(),
        partition: PredictionPartition::Validation,
        prediction_level: PredictionLevel::Target,
        fold_ids: vec![
            FoldId::new("fold:0").unwrap(),
            FoldId::new("fold:1").unwrap(),
        ],
        unit_ids: vec![target_a.clone(), target_b.clone()],
        sample_ids: Vec::new(),
        prediction_width: 1,
        target_names: vec!["y".to_string()],
    };
    let aggregated_blocks = vec![
        AggregatedPredictionBlock {
            prediction_id: Some("prediction:model:base.target.fold0".to_string()),
            producer_node: requirement.producer_node.clone(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            level: PredictionLevel::Target,
            unit_ids: vec![target_a],
            values: vec![vec![0.5]],
            target_names: vec!["y".to_string()],
        },
        AggregatedPredictionBlock {
            prediction_id: Some("prediction:model:base.target.fold1".to_string()),
            producer_node: requirement.producer_node.clone(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:1").unwrap()),
            level: PredictionLevel::Target,
            unit_ids: vec![target_b],
            values: vec![vec![0.7]],
            target_names: vec!["y".to_string()],
        },
    ];
    let cache = build_aggregated_prediction_cache_record(&requirement, &aggregated_blocks).unwrap();
    let payload =
        build_aggregated_prediction_cache_payload(&requirement, &aggregated_blocks).unwrap();
    let bundle = build_execution_bundle_with_prediction_contracts(
        BundleId::new("bundle:aggregated.prediction.cache").unwrap(),
        &plan,
        Some(plan.variants[0].variant_id.clone()),
        BTreeMap::new(),
        Vec::new(),
        vec![requirement.clone()],
        vec![cache.clone()],
    )
    .unwrap();
    let payload_set = BundlePredictionCachePayloadSet {
        bundle_id: bundle.bundle_id.clone(),
        schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
        caches: vec![payload.clone()],
    };

    let in_memory =
        InMemoryPredictionCacheStore::from_payloads(&bundle, payload_set.clone()).unwrap();
    assert!(in_memory.load_blocks(&requirement.key()).is_err());
    assert_eq!(
        in_memory
            .load_aggregated_blocks(&requirement.key())
            .unwrap(),
        aggregated_blocks
    );
    let handle = in_memory
        .materialize(&PredictionCacheMaterializationRequest {
            run_id: RunId::new("run:oof.edge.aggregated.cache.store.replay").unwrap(),
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Refit,
            variant_id: bundle.selected_variant_id.clone(),
            requirement: requirement.clone(),
            cache: cache.clone(),
            producer_controller_id: ControllerId::new("controller:model").unwrap(),
        })
        .unwrap();
    assert_eq!(handle.kind, HandleKind::Prediction);

    let columnar =
        ColumnarPredictionCacheStore::from_payloads(&bundle, payload_set.clone()).unwrap();
    assert_eq!(columnar.entry_count(), 1);
    let manifest = columnar.manifests();
    assert_eq!(manifest.len(), 1);
    assert_eq!(manifest[0].prediction_level, PredictionLevel::Target);
    assert_eq!(manifest[0].value_count, 2);
    assert!(columnar.load_blocks(&requirement.key()).is_err());
    assert_eq!(
        columnar.load_aggregated_blocks(&requirement.key()).unwrap(),
        aggregated_blocks
    );
    let columnar_handle = columnar
        .materialize(&PredictionCacheMaterializationRequest {
            run_id: RunId::new("run:oof.edge.aggregated.columnar.cache.store.replay").unwrap(),
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Refit,
            variant_id: bundle.selected_variant_id.clone(),
            requirement: requirement.clone(),
            cache: cache.clone(),
            producer_controller_id: ControllerId::new("controller:model").unwrap(),
        })
        .unwrap();
    assert_eq!(columnar_handle.kind, HandleKind::Prediction);

    let root = temp_prediction_cache_dir("dag_ml_aggregated_prediction_cache_store");
    let manifest =
        FilePredictionCacheStore::write_payload_set(&root, &bundle, &payload_set).unwrap();
    assert_eq!(manifest.caches[0].prediction_level, PredictionLevel::Target);
    assert_eq!(manifest.caches[0].unit_ids, requirement.unit_ids);
    let file_store = FilePredictionCacheStore::open(root.clone(), &bundle).unwrap();
    assert_eq!(
        file_store
            .load_aggregated_blocks(&requirement.key())
            .unwrap(),
        aggregated_blocks
    );
    let _ = fs::remove_dir_all(root);
}

#[test]
fn columnar_prediction_cache_block_round_trips_multi_target_rows() {
    let block = PredictionBlock {
        prediction_id: Some("pred:wide".to_string()),
        producer_node: NodeId::new("model:wide").unwrap(),
        partition: PredictionPartition::Validation,
        fold_id: Some(FoldId::new("fold:0").unwrap()),
        sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
        values: vec![vec![1.0, 10.0], vec![2.0, 20.0]],
        target_names: vec!["y0".to_string(), "y1".to_string()],
    };

    let columnar = ColumnarPredictionCacheBlock::from_prediction_block(&block).unwrap();
    assert_eq!(columnar.width, 2);
    assert_eq!(columnar.row_count(), 2);
    assert_eq!(columnar.value_count(), 4);
    assert_eq!(columnar.columns, vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
    assert_eq!(columnar.to_prediction_block().unwrap(), block);
}

#[test]
fn columnar_prediction_cache_block_round_trips_aggregated_units() {
    let block = AggregatedPredictionBlock {
        prediction_id: Some("pred:target".to_string()),
        producer_node: NodeId::new("model:target").unwrap(),
        partition: PredictionPartition::Validation,
        fold_id: Some(FoldId::new("fold:0").unwrap()),
        level: PredictionLevel::Target,
        unit_ids: vec![
            PredictionUnitId::Target(TargetId::new("target:a").unwrap()),
            PredictionUnitId::Target(TargetId::new("target:b").unwrap()),
        ],
        values: vec![vec![1.0, 10.0], vec![2.0, 20.0]],
        target_names: vec!["y0".to_string(), "y1".to_string()],
    };

    let columnar = ColumnarPredictionCacheBlock::from_aggregated_prediction_block(&block).unwrap();
    assert_eq!(columnar.prediction_level, PredictionLevel::Target);
    assert_eq!(columnar.row_count(), 2);
    assert_eq!(columnar.value_count(), 4);
    assert_eq!(columnar.columns, vec![vec![1.0, 2.0], vec![10.0, 20.0]]);
    assert!(columnar.to_prediction_block().is_err());
    assert_eq!(columnar.to_aggregated_prediction_block().unwrap(), block);
}

#[test]
fn columnar_prediction_cache_store_loads_and_materializes_oof_payloads() {
    let plan = build_execution_plan(
        "plan:oof.edge.columnar.cache.store",
        oof_edge_graph(),
        oof_edge_campaign(),
        &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
    )
    .unwrap();
    let fit_controllers = oof_edge_runtime_controllers(
        Some(PredictionPartition::Validation),
        OofSampleMode::Aligned,
    );
    let mut ctx = RunContext::new(
        RunId::new("run:oof.edge.columnar.cache.store").unwrap(),
        Some(11),
    );
    SequentialScheduler
        .execute_campaign_phase(&plan, &fit_controllers, &mut ctx, Phase::FitCv)
        .unwrap();

    let requirement = BundlePredictionRequirement {
        producer_node: NodeId::new("model:base").unwrap(),
        source_port: "pred".to_string(),
        consumer_node: NodeId::new("model:meta").unwrap(),
        target_port: "pred".to_string(),
        partition: PredictionPartition::Validation,
        prediction_level: PredictionLevel::Sample,
        fold_ids: vec![
            FoldId::new("fold:0").unwrap(),
            FoldId::new("fold:1").unwrap(),
        ],
        unit_ids: Vec::new(),
        sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
        prediction_width: 1,
        target_names: vec!["y".to_string()],
    };
    let cache = build_prediction_cache_record(&requirement, ctx.prediction_store.blocks()).unwrap();
    let payload =
        build_prediction_cache_payload(&requirement, ctx.prediction_store.blocks()).unwrap();
    let bundle = build_execution_bundle_with_prediction_contracts(
        BundleId::new("bundle:oof.edge.columnar.cache.store").unwrap(),
        &plan,
        Some(plan.variants[0].variant_id.clone()),
        BTreeMap::new(),
        Vec::new(),
        vec![requirement.clone()],
        vec![cache.clone()],
    )
    .unwrap();
    let payload_set = BundlePredictionCachePayloadSet {
        bundle_id: bundle.bundle_id.clone(),
        schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
        caches: vec![payload],
    };
    let store = ColumnarPredictionCacheStore::from_payloads(&bundle, payload_set).unwrap();
    assert_eq!(store.entry_count(), 1);
    let manifest = store.manifests();
    assert_eq!(manifest.len(), 1);
    assert_eq!(manifest[0].requirement_key, requirement.key());
    assert_eq!(manifest[0].prediction_level, PredictionLevel::Sample);
    assert_eq!(manifest[0].value_count, 2);
    assert_eq!(manifest[0].estimated_value_bytes, 16);
    assert_eq!(store.load_blocks(&requirement.key()).unwrap().len(), 2);

    let handle = store
        .materialize(&PredictionCacheMaterializationRequest {
            run_id: RunId::new("run:oof.edge.columnar.cache.store.replay").unwrap(),
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Refit,
            variant_id: bundle.selected_variant_id.clone(),
            requirement: requirement.clone(),
            cache,
            producer_controller_id: ControllerId::new("controller:model").unwrap(),
        })
        .unwrap();
    assert_eq!(handle.kind, HandleKind::Prediction);
    assert_eq!(
        handle.owner_controller,
        ControllerId::new("controller:model").unwrap()
    );
    let records = store.materialization_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].requirement_key, requirement.key());
    assert_eq!(records[0].handle, handle);
}

#[test]
fn file_prediction_cache_store_round_trips_oof_payloads_and_detects_tampering() {
    let plan = build_execution_plan(
        "plan:oof.edge.file.cache.store",
        oof_edge_graph(),
        oof_edge_campaign(),
        &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
    )
    .unwrap();
    let fit_controllers = oof_edge_runtime_controllers(
        Some(PredictionPartition::Validation),
        OofSampleMode::Aligned,
    );
    let mut ctx = RunContext::new(
        RunId::new("run:oof.edge.file.cache.store").unwrap(),
        Some(11),
    );
    SequentialScheduler
        .execute_campaign_phase(&plan, &fit_controllers, &mut ctx, Phase::FitCv)
        .unwrap();

    let requirement = BundlePredictionRequirement {
        producer_node: NodeId::new("model:base").unwrap(),
        source_port: "pred".to_string(),
        consumer_node: NodeId::new("model:meta").unwrap(),
        target_port: "pred".to_string(),
        partition: PredictionPartition::Validation,
        prediction_level: PredictionLevel::Sample,
        fold_ids: vec![
            FoldId::new("fold:0").unwrap(),
            FoldId::new("fold:1").unwrap(),
        ],
        unit_ids: Vec::new(),
        sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
        prediction_width: 1,
        target_names: vec!["y".to_string()],
    };
    let cache = build_prediction_cache_record(&requirement, ctx.prediction_store.blocks()).unwrap();
    let payload =
        build_prediction_cache_payload(&requirement, ctx.prediction_store.blocks()).unwrap();
    let bundle = build_execution_bundle_with_prediction_contracts(
        BundleId::new("bundle:oof.edge.file.cache.store").unwrap(),
        &plan,
        Some(plan.variants[0].variant_id.clone()),
        BTreeMap::new(),
        Vec::new(),
        vec![requirement.clone()],
        vec![cache.clone()],
    )
    .unwrap();
    let payload_set = BundlePredictionCachePayloadSet {
        bundle_id: bundle.bundle_id.clone(),
        schema_version: PREDICTION_CACHE_PAYLOAD_SCHEMA_VERSION,
        caches: vec![payload],
    };
    let root = temp_prediction_cache_dir("dag_ml_file_prediction_cache_store");

    let manifest =
        FilePredictionCacheStore::write_payload_set(&root, &bundle, &payload_set).unwrap();
    assert_eq!(manifest.caches.len(), 1);
    assert_eq!(manifest.caches[0].prediction_level, PredictionLevel::Sample);
    assert!(root.join(FILE_PREDICTION_CACHE_MANIFEST_FILE).exists());
    assert!(root.join(&manifest.caches[0].file_name).exists());

    let store = FilePredictionCacheStore::open(root.clone(), &bundle).unwrap();
    assert_eq!(store.manifest().caches, manifest.caches);
    assert_eq!(store.load_blocks(&requirement.key()).unwrap().len(), 2);
    let handle = store
        .materialize(&PredictionCacheMaterializationRequest {
            run_id: RunId::new("run:oof.edge.file.cache.store.replay").unwrap(),
            bundle_id: bundle.bundle_id.clone(),
            phase: Phase::Refit,
            variant_id: bundle.selected_variant_id.clone(),
            requirement: requirement.clone(),
            cache: cache.clone(),
            producer_controller_id: ControllerId::new("controller:model").unwrap(),
        })
        .unwrap();
    assert_eq!(handle.kind, HandleKind::Prediction);
    assert_eq!(store.materialization_records().len(), 1);

    let payload_path = root.join(&manifest.caches[0].file_name);
    let mut tampered: serde_json::Value =
        serde_json::from_slice(&fs::read(&payload_path).unwrap()).unwrap();
    tampered["blocks"][0]["values"][0][0] = json!(123456.0);
    fs::write(&payload_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
    let err = store.load_blocks(&requirement.key()).unwrap_err();
    assert!(
        err.to_string().contains("content fingerprint"),
        "unexpected tamper error: {err}"
    );

    let _ = fs::remove_dir_all(root);
}

fn portable_artifact_bundle(plan: &ExecutionPlan) -> crate::bundle::ExecutionBundle {
    let model_plan = plan
        .node_plans
        .get(&NodeId::new("model:base").unwrap())
        .unwrap();
    let content_fingerprint = "a".repeat(64);
    build_execution_bundle(
        crate::ids::BundleId::new("bundle:artifact.manifest").unwrap(),
        plan,
        Some(plan.variants[0].variant_id.clone()),
        BTreeMap::new(),
        vec![RefitArtifactRecord {
            node_id: model_plan.node_id.clone(),
            controller_id: model_plan.controller_id.clone(),
            artifact: ArtifactRef {
                id: ArtifactId::new("artifact:model:base:refit").unwrap(),
                kind: "mock_model".to_string(),
                controller_id: model_plan.controller_id.clone(),
                backend: Some(ArtifactBackend::Joblib),
                uri: Some(format!("artifacts/{content_fingerprint}.joblib")),
                content_fingerprint: Some(content_fingerprint),
                size_bytes: Some(128),
                plugin: Some("dagml.mock".to_string()),
                plugin_version: Some("1.0.0".to_string()),
            },
            params_fingerprint: model_plan.params_fingerprint.clone(),
            data_requirement_keys: vec!["model:base.x".to_string()],
            prediction_requirement_keys: Vec::new(),
        }],
    )
    .unwrap()
}

fn portable_artifact_bundle_with_payload(
    plan: &ExecutionPlan,
    payload: &[u8],
) -> crate::bundle::ExecutionBundle {
    let mut bundle = portable_artifact_bundle(plan);
    let content_fingerprint = sha256_bytes_hex(payload);
    let artifact = &mut bundle.refit_artifacts[0].artifact;
    artifact.uri = Some(format!("artifacts/{content_fingerprint}.joblib"));
    artifact.content_fingerprint = Some(content_fingerprint);
    artifact.size_bytes = Some(payload.len() as u64);
    bundle.validate().unwrap();
    bundle
}

fn write_artifact_payload(root: &Path, bundle: &ExecutionBundle, payload: &[u8]) -> PathBuf {
    let uri = bundle.refit_artifacts[0].artifact.uri.as_deref().unwrap();
    let path = root.join(uri);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, payload).unwrap();
    path
}

#[test]
fn artifact_ref_validate_portable_rejects_unsafe_uris_and_legacy() {
    let content_fingerprint = "c".repeat(64);
    let base = ArtifactRef {
        id: ArtifactId::new("artifact:model:portable").unwrap(),
        kind: "model".to_string(),
        controller_id: ControllerId::new("controller:sklearn").unwrap(),
        backend: Some(ArtifactBackend::Joblib),
        uri: Some(format!("artifacts/{content_fingerprint}.joblib")),
        content_fingerprint: Some(content_fingerprint),
        size_bytes: Some(4096),
        plugin: Some("dagml.sklearn".to_string()),
        plugin_version: Some("1.0.0".to_string()),
    };
    base.validate_portable().unwrap();

    // Legacy artifact: still passes `validate` but is refused as non-portable.
    let legacy = ArtifactRef {
        backend: None,
        uri: None,
        content_fingerprint: None,
        ..base.clone()
    };
    legacy.validate().unwrap();
    assert!(legacy
        .validate_portable()
        .unwrap_err()
        .to_string()
        .contains("not portable"));

    let mut absolute = base.clone();
    absolute.uri = Some("/etc/passwd".to_string());
    assert!(absolute
        .validate_portable()
        .unwrap_err()
        .to_string()
        .contains("must be a relative path"));

    let mut traversal = base.clone();
    traversal.uri = Some("artifacts/../../secret.joblib".to_string());
    assert!(traversal
        .validate_portable()
        .unwrap_err()
        .to_string()
        .contains("`..`"));

    let mut drive = base.clone();
    drive.uri = Some("C:\\models\\model.joblib".to_string());
    assert!(drive
        .validate_portable()
        .unwrap_err()
        .to_string()
        .contains("must be a relative path"));

    // URI schemes and any other colon in the leading path segment are
    // rejected: a strictly relative artifact path never carries a scheme.
    for scheme_uri in [
        "http://example.com/model.joblib",
        "s3://bucket/model.joblib",
        "file:///models/model.joblib",
        "weird:thing/model.joblib",
    ] {
        let mut scheme = base.clone();
        scheme.uri = Some(scheme_uri.to_string());
        let err = scheme.validate_portable().unwrap_err().to_string();
        assert!(
            err.contains("first path segment"),
            "unexpected scheme error for `{scheme_uri}`: {err}"
        );
    }

    // A colon outside the first segment is allowed (not a scheme/drive).
    let mut later_colon = base;
    later_colon.uri = Some("artifacts/model:v1.joblib".to_string());
    later_colon.validate_portable().unwrap();
}

#[test]
fn file_artifact_manifest_round_trips_portable_artifacts() {
    let plan = fixture_plan("plan:artifact.manifest.round.trip");
    let bundle = portable_artifact_bundle(&plan);
    let root = temp_prediction_cache_dir("dag_ml_file_artifact_manifest");

    let manifest = FileArtifactManifestStore::write(&root, &bundle).unwrap();
    assert_eq!(
        manifest.schema_version,
        FILE_ARTIFACT_MANIFEST_SCHEMA_VERSION
    );
    assert_eq!(manifest.artifacts.len(), 1);
    assert_eq!(
        manifest.artifacts[0].artifact.id,
        ArtifactId::new("artifact:model:base:refit").unwrap()
    );
    assert_eq!(
        manifest.artifacts[0].artifact.backend,
        Some(ArtifactBackend::Joblib)
    );
    assert_eq!(
        manifest.artifacts[0].node_id,
        bundle.refit_artifacts[0].node_id
    );
    assert!(root.join(FILE_ARTIFACT_MANIFEST_FILE).exists());

    let store = FileArtifactManifestStore::open(root.clone(), &bundle).unwrap();
    assert_eq!(store.root(), root.as_path());
    assert_eq!(store.manifest().bundle_id, bundle.bundle_id);
    assert_eq!(store.manifest().artifacts, manifest.artifacts);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn file_artifact_manifest_refuses_legacy_non_portable_artifacts() {
    let plan = fixture_plan("plan:artifact.manifest.legacy");
    // `replay_bundle` carries a legacy artifact (no backend/uri/content fingerprint).
    let bundle = replay_bundle(&plan);
    let root = temp_prediction_cache_dir("dag_ml_file_artifact_manifest_legacy");

    let err = FileArtifactManifestStore::write(&root, &bundle).unwrap_err();
    assert!(
        err.to_string().contains("not portable"),
        "unexpected legacy error: {err}"
    );
    assert!(!root.join(FILE_ARTIFACT_MANIFEST_FILE).exists());

    let _ = fs::remove_dir_all(root);
}

#[test]
fn file_artifact_manifest_open_refuses_tampered_entries() {
    let plan = fixture_plan("plan:artifact.manifest.tampered");
    let bundle = portable_artifact_bundle(&plan);
    let root = temp_prediction_cache_dir("dag_ml_file_artifact_manifest_tampered");

    FileArtifactManifestStore::write(&root, &bundle).unwrap();
    let manifest_path = root.join(FILE_ARTIFACT_MANIFEST_FILE);
    let mut tampered: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let tampered_fingerprint = "b".repeat(64);
    tampered["artifacts"][0]["params_fingerprint"] = json!(tampered_fingerprint);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&tampered).unwrap(),
    )
    .unwrap();

    let err = FileArtifactManifestStore::open(root.clone(), &bundle).unwrap_err();
    assert!(
        err.to_string()
            .contains("does not match bundle refit artifact"),
        "unexpected tamper error: {err}"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn file_artifact_payload_store_validates_payloads_and_materializes_handles() {
    let plan = fixture_plan("plan:artifact.payload.round.trip");
    let payload = b"portable dag-ml artifact payload\n";
    let bundle = portable_artifact_bundle_with_payload(&plan, payload);
    let source_root = temp_prediction_cache_dir("dag_ml_file_artifact_payload_source");
    let store_root = temp_prediction_cache_dir("dag_ml_file_artifact_payload_store");
    let source_path = write_artifact_payload(&source_root, &bundle, payload);

    let store =
        FileArtifactPayloadStore::write_from_source(&store_root, &source_root, &bundle).unwrap();
    assert_eq!(store.root(), store_root.as_path());
    assert_eq!(store.payload_count(), 1);
    assert!(store_root
        .join(bundle.refit_artifacts[0].artifact.uri.as_deref().unwrap())
        .exists());
    assert!(source_path.exists());
    assert_eq!(store.manifest().bundle_id, bundle.bundle_id);

    let artifact = &bundle.refit_artifacts[0];
    let handle = store
        .materialize(&ArtifactMaterializationRequest {
            run_id: RunId::new("run:artifact.payload.materialize").unwrap(),
            bundle_id: bundle.bundle_id.clone(),
            node_id: artifact.node_id.clone(),
            phase: Phase::Predict,
            variant_id: bundle.selected_variant_id.clone(),
            controller_id: artifact.controller_id.clone(),
            artifact: artifact.artifact.clone(),
            params_fingerprint: artifact.params_fingerprint.clone(),
        })
        .unwrap();
    assert_eq!(handle.kind, HandleKind::Artifact);
    assert_eq!(handle.owner_controller, artifact.controller_id);
    let records = store.materialization_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].artifact_id, artifact.artifact.id);
    assert_eq!(records[0].size_bytes, payload.len() as u64);
    assert_eq!(
        records[0].content_fingerprint,
        artifact.artifact.content_fingerprint.clone().unwrap()
    );

    let reopened = FileArtifactPayloadStore::open(store_root.clone(), &bundle).unwrap();
    reopened.validate_payloads().unwrap();

    let _ = fs::remove_dir_all(source_root);
    let _ = fs::remove_dir_all(store_root);
}

#[test]
fn file_artifact_payload_store_refuses_tampered_payloads() {
    let plan = fixture_plan("plan:artifact.payload.tampered");
    let payload = b"portable dag-ml artifact payload\n";
    let bundle = portable_artifact_bundle_with_payload(&plan, payload);
    let source_root = temp_prediction_cache_dir("dag_ml_file_artifact_payload_source_tamper");
    let store_root = temp_prediction_cache_dir("dag_ml_file_artifact_payload_store_tamper");
    write_artifact_payload(&source_root, &bundle, payload);
    FileArtifactPayloadStore::write_from_source(&store_root, &source_root, &bundle).unwrap();

    let payload_path = store_root.join(bundle.refit_artifacts[0].artifact.uri.as_deref().unwrap());
    fs::write(&payload_path, vec![b'x'; payload.len()]).unwrap();
    let err = FileArtifactPayloadStore::open(store_root.clone(), &bundle).unwrap_err();
    assert!(
        err.to_string().contains("content fingerprint mismatch"),
        "unexpected tamper error: {err}"
    );

    let _ = fs::remove_dir_all(source_root);
    let _ = fs::remove_dir_all(store_root);
}

#[test]
fn requires_oof_prediction_edge_refit_rejects_incomplete_oof_coverage() {
    let plan = build_execution_plan(
        "plan:oof.edge.refit.incomplete",
        oof_edge_graph(),
        oof_edge_campaign(),
        &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
    )
    .unwrap();
    let mut ctx = RunContext::new(
        RunId::new("run:oof.edge.refit.incomplete").unwrap(),
        Some(11),
    );
    ctx.prediction_store
        .append(PredictionBlock {
            prediction_id: Some("pred:model:base:fold0".to_string()),
            producer_node: NodeId::new("model:base").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            sample_ids: vec![SampleId::new("s1").unwrap()],
            values: vec![vec![0.5]],
            target_names: vec!["y".to_string()],
        })
        .unwrap();
    let controllers = oof_edge_runtime_controllers(None, OofSampleMode::Aligned);

    let error = SequentialScheduler
        .execute_campaign_phase(&plan, &controllers, &mut ctx, Phase::Refit)
        .unwrap_err()
        .to_string();

    assert!(error.contains("do not cover the refit sample universe"));
}

#[test]
fn data_bindings_require_runtime_provider_and_materialize_handles() {
    let model_id = NodeId::new("model:pls").unwrap();
    let plan = build_execution_plan(
        "plan:data",
        simple_graph(),
        CampaignSpec {
            id: "campaign:data".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::from([(model_id.clone(), vec![data_binding(&model_id)])]),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let controllers = runtime_controllers();
    let mut ctx = RunContext::new(RunId::new("run:data").unwrap(), Some(11));

    assert!(SequentialScheduler
        .execute_phase(&plan, &controllers, &mut ctx, Phase::FitCv)
        .is_err());

    let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
    ))
    .unwrap();
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data.provider").unwrap(),
        envelope,
    )
    .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:data.provider").unwrap(), Some(11));
    let results = SequentialScheduler
        .execute_phase_with_data_provider(&plan, &controllers, &provider, &mut ctx, Phase::FitCv)
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(provider.handle_records().len(), 1);
    assert_eq!(provider.view_records().len(), 1);
    assert_eq!(provider.handle_records()[0].input_name, "x");
    assert_eq!(provider.handle_records()[0].relation_record_count, Some(4));
    assert_eq!(provider.view_records()[0].handle.kind, HandleKind::DataView);
    assert_eq!(
        provider.view_records()[0].parent_handle,
        provider.handle_records()[0].handle
    );
}

#[test]
fn campaign_data_bindings_create_fold_train_views() {
    let model_id = NodeId::new("model:pls").unwrap();
    let plan = build_execution_plan(
        "plan:data.folds",
        simple_graph(),
        CampaignSpec {
            id: "campaign:data.folds".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: Some(SplitInvocation {
                id: "split:outer".to_string(),
                controller_id: None,
                leakage_policy: Default::default(),
                params: BTreeMap::new(),
                fold_set: Some(two_fold_set()),
            }),
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::from([(
                model_id,
                vec![data_binding(&NodeId::new("model:pls").unwrap())],
            )]),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
    ))
    .unwrap();
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data.provider").unwrap(),
        envelope,
    )
    .unwrap();
    let controllers = runtime_controllers();
    let mut ctx = RunContext::new(RunId::new("run:data.folds").unwrap(), Some(11));

    let results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap();

    assert_eq!(results.len(), 4);
    assert_eq!(provider.handle_records().len(), 2);
    let views = provider.view_records();
    assert_eq!(views.len(), 4);
    assert!(views
        .iter()
        .all(|view| view.handle.kind == HandleKind::DataView));
    let train_views = views
        .iter()
        .filter(|view| view.view.partition == DataRequestPartition::FoldTrain)
        .collect::<Vec<_>>();
    let validation_views = views
        .iter()
        .filter(|view| view.view.partition == DataRequestPartition::FoldValidation)
        .collect::<Vec<_>>();
    assert_eq!(train_views.len(), 2);
    assert_eq!(validation_views.len(), 2);
    assert_eq!(
        train_views[0].view.sample_ids,
        Some(vec![SampleId::new("s2").unwrap()])
    );
    assert_eq!(
        validation_views[0].view.sample_ids,
        Some(vec![SampleId::new("s1").unwrap()])
    );
    assert_eq!(
        train_views[1].view.sample_ids,
        Some(vec![SampleId::new("s1").unwrap()])
    );
    assert_eq!(
        validation_views[1].view.sample_ids,
        Some(vec![SampleId::new("s2").unwrap()])
    );
}

#[test]
fn data_edges_propagate_fold_views_from_data_producing_nodes() {
    let augment_id = NodeId::new("augment:noise").unwrap();
    let model_id = NodeId::new("model:branch").unwrap();
    let before_feature_schema = "a".repeat(64);
    let after_feature_schema = "b".repeat(64);
    let shape_plan = DataModelShapePlan {
        node_id: augment_id.clone(),
        input_granularity: Granularity::Sample,
        target_granularity: Granularity::Sample,
        fit_rows: FitBoundary::FoldTrain,
        predict_rows: FitBoundary::FoldValidation,
        feature_namespace: Some("augmented.noise".to_string()),
        feature_schema_fingerprint: Some(before_feature_schema.clone()),
        target_space: "raw".to_string(),
        aggregation_policy: AggregationPolicy::default(),
        augmentation_policy: Default::default(),
        selection_policy: Default::default(),
    };
    let shape_plan_fingerprint = stable_json_fingerprint(&shape_plan).unwrap();
    let graph = GraphSpec {
        id: "g:data.edge.views".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![
            node(
                augment_id.as_str(),
                NodeKind::Augmentation,
                vec![port("x", PortKind::Data)],
                vec![port("x_out", PortKind::Data)],
            ),
            node(
                model_id.as_str(),
                NodeKind::Model,
                vec![port("x", PortKind::Data)],
                vec![port("oof", PortKind::Prediction)],
            ),
        ],
        edges: vec![EdgeSpec {
            source: PortRef {
                node_id: augment_id.clone(),
                port_name: "x_out".to_string(),
            },
            target: PortRef {
                node_id: model_id.clone(),
                port_name: "x".to_string(),
            },
            contract: EdgeContract {
                kind: PortKind::Data,
                representation: None,
                requires_oof: false,
                requires_fold_alignment: false,
                propagates_lineage: true,
            },
        }],
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    };
    let mut manifest_registry = ControllerRegistry::new();
    manifest_registry
        .register(controller_manifest(
            "controller:augmentation",
            NodeKind::Augmentation,
        ))
        .unwrap();
    manifest_registry
        .register(controller_manifest(
            "controller:model.probe",
            NodeKind::Model,
        ))
        .unwrap();
    let plan = build_execution_plan(
        "plan:data.edge.views",
        graph,
        CampaignSpec {
            id: "campaign:data.edge.views".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: Some(SplitInvocation {
                id: "split:outer".to_string(),
                controller_id: None,
                leakage_policy: Default::default(),
                params: BTreeMap::new(),
                fold_set: Some(two_fold_set()),
            }),
            generation: Default::default(),
            shape_plans: BTreeMap::from([(augment_id.clone(), shape_plan)]),
            data_bindings: BTreeMap::from([(augment_id.clone(), vec![data_binding(&augment_id)])]),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifest_registry,
    )
    .unwrap();
    let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
    ))
    .unwrap();
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data.provider").unwrap(),
        envelope,
    )
    .unwrap();
    let observed_views = Arc::new(Mutex::new(Vec::new()));
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(ShapeDataController {
            id: ControllerId::new("controller:augmentation").unwrap(),
            handle: 3,
            before_feature_schema: before_feature_schema.clone(),
            after_feature_schema: after_feature_schema.clone(),
        }))
        .unwrap();
    controllers
        .register(Box::new(DataViewProbeController {
            id: ControllerId::new("controller:model.probe").unwrap(),
            observed_views: observed_views.clone(),
            prediction_sample_ids: None,
        }))
        .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:data.edge.views").unwrap(), Some(11));

    let results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap();

    assert_eq!(results.len(), 4);
    assert_eq!(provider.view_records().len(), 4);
    let observed_views = observed_views.lock().unwrap();
    assert_eq!(observed_views.len(), 2);
    for views in observed_views.iter() {
        let primary = views.get("data:x").expect("primary propagated data view");
        let validation = views
            .get("data:x:validation")
            .expect("validation propagated data view");
        for view in [primary, validation] {
            let provenance = view
                .output_provenance()
                .unwrap()
                .expect("output data provenance metadata");
            assert_eq!(
                provenance.producer_node,
                NodeId::new("augment:noise").unwrap()
            );
            assert_eq!(provenance.producer_port, "x_out");
            assert_eq!(
                provenance.shape_plan_fingerprint,
                Some(shape_plan_fingerprint.clone())
            );
            assert_eq!(
                provenance.feature_schema_fingerprint,
                Some(after_feature_schema.clone())
            );
            assert_eq!(provenance.shape_deltas.len(), 1);
        }
    }
    let samples_by_fold = ctx
        .prediction_store
        .blocks()
        .iter()
        .filter(|block| block.producer_node == model_id)
        .map(|block| {
            (
                block.fold_id.as_ref().unwrap().to_string(),
                block.sample_ids.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        samples_by_fold["fold:0"],
        vec![SampleId::new("s1").unwrap()]
    );
    assert_eq!(
        samples_by_fold["fold:1"],
        vec![SampleId::new("s2").unwrap()]
    );

    let mut bad_controllers = RuntimeControllerRegistry::new();
    bad_controllers
        .register(Box::new(ShapeDataController {
            id: ControllerId::new("controller:augmentation").unwrap(),
            handle: 5,
            before_feature_schema,
            after_feature_schema,
        }))
        .unwrap();
    bad_controllers
        .register(Box::new(DataViewProbeController {
            id: ControllerId::new("controller:model.probe").unwrap(),
            observed_views: Arc::new(Mutex::new(Vec::new())),
            prediction_sample_ids: Some(vec![SampleId::new("s-outside").unwrap()]),
        }))
        .unwrap();
    let mut bad_ctx = RunContext::new(
        RunId::new("run:data.edge.views.bad-prediction").unwrap(),
        Some(11),
    );
    let error = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &bad_controllers,
            &provider,
            &mut bad_ctx,
            Phase::FitCv,
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("outside its validation view"),
        "unexpected propagated-view validation error: {error}"
    );
}

#[test]
fn data_provider_view_validates_typed_output_provenance() {
    let producer = NodeId::new("augment:noise").unwrap();
    let before_feature_schema = "a".repeat(64);
    let after_feature_schema = "b".repeat(64);
    let provenance = DataOutputProvenance {
        schema_version: DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION,
        producer_node: producer.clone(),
        producer_port: "x_out".to_string(),
        producer_phase: Phase::FitCv,
        variant_id: Some(VariantId::new("variant:base").unwrap()),
        fold_id: Some(FoldId::new("fold:0").unwrap()),
        shape_plan_fingerprint: Some("c".repeat(64)),
        aggregation_policy_fingerprint: Some("d".repeat(64)),
        feature_namespace: Some("augmented.noise".to_string()),
        feature_schema_fingerprint: Some(after_feature_schema.clone()),
        shape_deltas: vec![ShapeDelta {
            node_id: producer.clone(),
            kind: ShapeDeltaKind::Feature,
            before_fingerprint: before_feature_schema,
            after_fingerprint: after_feature_schema,
            metadata: BTreeMap::new(),
        }],
    };
    let mut view = DataProviderViewSpec {
        sample_ids: Some(vec![SampleId::new("s1").unwrap()]),
        partition: DataRequestPartition::FoldTrain,
        fold_id: Some(FoldId::new("fold:0").unwrap()),
        source_ids: None,
        columns: None,
        include_augmented: true,
        include_excluded: false,
        branch_view: None,
        extra: BTreeMap::from([(
            DATA_OUTPUT_PROVENANCE_KEY.to_string(),
            serde_json::to_value(&provenance).unwrap(),
        )]),
    };

    assert_eq!(view.output_provenance().unwrap(), Some(provenance.clone()));
    view.validate().unwrap();

    let mut empty_port = provenance.clone();
    empty_port.producer_port.clear();
    view.extra.insert(
        DATA_OUTPUT_PROVENANCE_KEY.to_string(),
        serde_json::to_value(empty_port).unwrap(),
    );
    let error = view.validate().unwrap_err().to_string();
    assert!(
        error.contains("empty producer_port"),
        "unexpected empty-port provenance error: {error}"
    );

    let mut wrong_delta_node = provenance.clone();
    wrong_delta_node.shape_deltas[0].node_id = NodeId::new("augment:other").unwrap();
    view.extra.insert(
        DATA_OUTPUT_PROVENANCE_KEY.to_string(),
        serde_json::to_value(wrong_delta_node).unwrap(),
    );
    let error = view.validate().unwrap_err().to_string();
    assert!(
        error.contains("contains shape delta"),
        "unexpected wrong-delta-node provenance error: {error}"
    );

    let mut wrong_feature_fingerprint = provenance.clone();
    wrong_feature_fingerprint.feature_schema_fingerprint = Some("e".repeat(64));
    view.extra.insert(
        DATA_OUTPUT_PROVENANCE_KEY.to_string(),
        serde_json::to_value(wrong_feature_fingerprint).unwrap(),
    );
    let error = view.validate().unwrap_err().to_string();
    assert!(
        error.contains("last feature delta"),
        "unexpected feature-fingerprint provenance error: {error}"
    );

    let mut unsupported_schema = provenance;
    unsupported_schema.schema_version = DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION + 1;
    view.extra.insert(
        DATA_OUTPUT_PROVENANCE_KEY.to_string(),
        serde_json::to_value(unsupported_schema).unwrap(),
    );
    let error = view.validate().unwrap_err().to_string();
    assert!(
        error.contains("unsupported schema_version"),
        "unexpected provenance schema-version error: {error}"
    );
}

#[test]
fn data_provider_view_spec_propagates_branch_view_validation() {
    use crate::data::{BranchViewMode, BranchViewPlan, DataViewSelector};

    let view = DataProviderViewSpec {
        sample_ids: None,
        partition: DataRequestPartition::FullTrain,
        fold_id: None,
        source_ids: None,
        columns: None,
        include_augmented: true,
        include_excluded: false,
        branch_view: Some(BranchViewPlan {
            view_id: "branch_view:nir_only".to_string(),
            branch_id: "branch:nir".to_string(),
            mode: BranchViewMode::BySource,
            selector: DataViewSelector {
                source_ids: vec!["nir".to_string()],
                ..Default::default()
            },
            allow_overlap: false,
            metadata: BTreeMap::new(),
        }),
        extra: BTreeMap::new(),
    };
    view.validate().unwrap();

    let invalid = DataProviderViewSpec {
        branch_view: Some(BranchViewPlan {
            view_id: "branch_view:bad".to_string(),
            branch_id: "branch:bad".to_string(),
            mode: BranchViewMode::BySource,
            selector: DataViewSelector::default(),
            allow_overlap: false,
            metadata: BTreeMap::new(),
        }),
        ..view
    };
    let error = invalid.validate().unwrap_err().to_string();
    assert!(
        error.contains("selector must constrain source_ids, metadata, tags or filter"),
        "unexpected: {error}"
    );
}

#[test]
fn scheduler_extracts_branch_view_from_node_metadata() {
    use crate::data::{BranchViewMode, BranchViewPlan, DataViewSelector};
    use crate::graph::NodeSpec;

    let plan_with_branch = BranchViewPlan {
        view_id: "branch_view:nir_only".to_string(),
        branch_id: "branch:nir".to_string(),
        mode: BranchViewMode::BySource,
        selector: DataViewSelector {
            source_ids: vec!["nir".to_string()],
            ..Default::default()
        },
        allow_overlap: false,
        metadata: BTreeMap::new(),
    };

    let node_id = NodeId::new("model:branched").unwrap();
    let mut node_spec_metadata = BTreeMap::new();
    node_spec_metadata.insert(
        "dsl_branch_view_plan".to_string(),
        serde_json::to_value(&plan_with_branch).unwrap(),
    );
    let node_spec = NodeSpec {
        id: node_id.clone(),
        kind: crate::graph::NodeKind::Model,
        operator: None,
        params: BTreeMap::new(),
        ports: Default::default(),
        metadata: node_spec_metadata,
        seed_label: None,
    };

    let other_node = NodeSpec {
        id: NodeId::new("model:plain").unwrap(),
        kind: crate::graph::NodeKind::Model,
        operator: None,
        params: BTreeMap::new(),
        ports: Default::default(),
        metadata: BTreeMap::new(),
        seed_label: None,
    };

    let graph = crate::graph::GraphSpec {
        id: "g".to_string(),
        interface: Default::default(),
        nodes: vec![node_spec, other_node],
        edges: Vec::new(),
        metadata: BTreeMap::new(),
        search_space_fingerprint: None,
    };
    let plan = ExecutionPlan {
        id: "plan:test".to_string(),
        graph_plan: crate::plan::GraphPlan {
            graph,
            topological_order: vec![
                node_id.clone(),
                NodeId::new("model:plain").unwrap(),
            ],
            parallel_levels: Vec::new(),
        },
        campaign: crate::plan::CampaignSpec {
            id: "campaign:test".to_string(),
            root_seed: None,
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        node_plans: BTreeMap::new(),
        controller_manifests: BTreeMap::new(),
        variants: Vec::new(),
        fold_set: None,
        graph_fingerprint: String::new(),
        campaign_fingerprint: String::new(),
        controller_fingerprint: String::new(),
    };

    let resolved = super::branch_view_from_node_metadata(&plan, &node_id).unwrap();
    assert_eq!(resolved.as_ref(), Some(&plan_with_branch));

    let plain_resolved =
        super::branch_view_from_node_metadata(&plan, &NodeId::new("model:plain").unwrap())
            .unwrap();
    assert_eq!(plain_resolved, None);

    let missing_resolved =
        super::branch_view_from_node_metadata(&plan, &NodeId::new("model:unknown").unwrap())
            .unwrap();
    assert_eq!(missing_resolved, None);
}

#[test]
fn published_data_output_provenance_schema_declares_current_version() {
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/contracts/data_output_provenance.schema.json"
    ))
    .unwrap();
    assert_eq!(
        schema["properties"]["schema_version"]["const"].as_u64(),
        Some(u64::from(DATA_OUTPUT_PROVENANCE_SCHEMA_VERSION))
    );
    assert_eq!(schema["$id"], DATA_OUTPUT_PROVENANCE_SCHEMA_ID);
    let required = schema["required"].as_array().unwrap();
    assert!(required
        .iter()
        .any(|field| field.as_str() == Some("schema_version")));
    assert!(required
        .iter()
        .any(|field| field.as_str() == Some("producer_node")));
}

#[test]
fn published_node_task_and_result_schemas_declare_current_contracts() {
    let task_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/contracts/node_task.schema.json"
    ))
    .unwrap();
    let result_schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/contracts/node_result.schema.json"
    ))
    .unwrap();

    assert_eq!(task_schema["$id"], NODE_TASK_SCHEMA_ID);
    assert_eq!(result_schema["$id"], NODE_RESULT_SCHEMA_ID);
    assert!(task_schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .any(|field| field.as_str() == Some("node_plan")));
    assert!(result_schema["required"]
        .as_array()
        .unwrap()
        .iter()
        .any(|field| field.as_str() == Some("lineage")));
}

#[test]
fn published_node_task_result_fixtures_validate_current_contract() {
    let task: NodeTask = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/runtime/node_task_transform_scale.json"
    ))
    .unwrap();
    let result: NodeResult = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/runtime/node_result_transform_scale.json"
    ))
    .unwrap();

    result.validate_for_task(&task).unwrap();
    assert_eq!(
        task.node_plan.node_id,
        NodeId::new("transform:scale").unwrap()
    );
    assert_eq!(result.outputs.len(), 1);
}

#[test]
fn campaign_data_bindings_require_unsafe_flags_for_full_train_cv_views() {
    let model_id = NodeId::new("model:pls").unwrap();
    let mut unsafe_binding = data_binding(&model_id);
    unsafe_binding.view_policy.fit_partition = DataRequestPartition::FullTrain;
    unsafe_binding.view_policy.unsafe_flags =
        BTreeSet::from([DataViewPolicy::ALLOW_FIT_CV_FULL_TRAIN_VIEW.to_string()]);

    let mut unsafe_campaign = oof_edge_campaign();
    unsafe_campaign.data_bindings =
        BTreeMap::from([(model_id.clone(), vec![unsafe_binding.clone()])]);
    let plan = build_execution_plan(
        "plan:data.full-train.unsafe",
        simple_graph(),
        unsafe_campaign,
        &manifests(),
    )
    .unwrap();

    let mut missing_flag = unsafe_binding;
    missing_flag.view_policy.unsafe_flags.clear();
    let mut invalid_campaign = oof_edge_campaign();
    invalid_campaign.data_bindings = BTreeMap::from([(model_id.clone(), vec![missing_flag])]);
    assert!(build_execution_plan(
        "plan:data.full-train.missing-flag",
        simple_graph(),
        invalid_campaign,
        &manifests(),
    )
    .is_err());

    let envelope: ExternalDataPlanEnvelope = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/data/coordinator_data_plan_envelope_sample12.json"
    ))
    .unwrap();
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data.provider").unwrap(),
        envelope,
    )
    .unwrap();
    let controllers = runtime_controllers();
    let mut ctx = RunContext::new(RunId::new("run:data.full-train.unsafe").unwrap(), Some(11));

    SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap();

    let full_train_ids = plan.fold_set.as_ref().unwrap().sample_ids.clone();
    let views = provider.view_records();
    let full_train_views = views
        .iter()
        .filter(|view| view.view.partition == DataRequestPartition::FullTrain)
        .collect::<Vec<_>>();
    let validation_views = views
        .iter()
        .filter(|view| view.view.partition == DataRequestPartition::FoldValidation)
        .collect::<Vec<_>>();
    assert_eq!(full_train_views.len(), 2);
    assert_eq!(validation_views.len(), 2);
    assert!(full_train_views.iter().all(|view| {
        view.view.sample_ids == Some(full_train_ids.clone())
            && view.view.fold_id.is_none()
            && view.view.extra["unsafe_flags"]
                .as_array()
                .unwrap()
                .iter()
                .any(|flag| flag.as_str() == Some(DataViewPolicy::ALLOW_FIT_CV_FULL_TRAIN_VIEW))
    }));
    assert!(validation_views
        .iter()
        .all(|view| !view.view.include_augmented));
}

#[test]
fn campaign_refit_data_bindings_create_full_train_views() {
    let plan = fixture_plan("plan:refit.views");
    let provider = replay_data_provider();
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(ReplayMockController {
            id: ControllerId::new("controller:transform.mock").unwrap(),
            handle: 11,
            require_artifact: false,
            emit_prediction: false,
            emit_refit_artifact: false,
        }))
        .unwrap();
    controllers
        .register(Box::new(ReplayMockController {
            id: ControllerId::new("controller:model.mock").unwrap(),
            handle: 22,
            require_artifact: false,
            emit_prediction: true,
            emit_refit_artifact: false,
        }))
        .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:refit.views").unwrap(), Some(11));
    ctx.variant_id = Some(plan.variants[0].variant_id.clone());

    let results = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &provider,
            &mut ctx,
            Phase::Refit,
        )
        .unwrap();

    assert!(!results.is_empty());
    let views = provider.view_records();
    assert_eq!(views.len(), 1);
    let full_train_ids = plan.fold_set.as_ref().unwrap().sample_ids.clone();
    assert!(views.iter().all(|view| {
        view.view.partition == DataRequestPartition::FullTrain
            && view.view.sample_ids == Some(full_train_ids.clone())
            && view.fold_id.is_none()
    }));
}

#[test]
fn campaign_refit_captures_emitted_artifact_handles() {
    let plan = fixture_plan("plan:refit.artifact.capture");
    let provider = replay_data_provider();
    let mut artifact_store = InMemoryArtifactStore::new();
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(ReplayMockController {
            id: ControllerId::new("controller:transform.mock").unwrap(),
            handle: 11,
            require_artifact: false,
            emit_prediction: false,
            emit_refit_artifact: false,
        }))
        .unwrap();
    controllers
        .register(Box::new(ReplayMockController {
            id: ControllerId::new("controller:model.mock").unwrap(),
            handle: 22,
            require_artifact: false,
            emit_prediction: true,
            emit_refit_artifact: true,
        }))
        .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:refit.artifact.capture").unwrap(), Some(11));
    ctx.variant_id = Some(plan.variants[0].variant_id.clone());

    let results = SequentialScheduler
        .execute_campaign_phase_with_data_provider_and_artifact_store(
            &plan,
            &controllers,
            &provider,
            &mut artifact_store,
            &mut ctx,
            Phase::Refit,
        )
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(
        results
            .iter()
            .filter(|result| !result.artifacts.is_empty())
            .count(),
        1
    );
    assert_eq!(artifact_store.len(), 1);
    let records = artifact_store.refit_artifacts();
    assert_eq!(records.len(), 1);
    let artifact = &records[0];
    artifact.validate().unwrap();
    assert_eq!(artifact.node_id.as_str(), "model:base");
    assert_eq!(artifact.controller_id.as_str(), "controller:model.mock");
    assert_eq!(artifact.artifact.id.as_str(), "artifact:model:base:refit");
    assert_eq!(artifact.data_requirement_keys, vec!["model:base.x"]);

    let handle = artifact_store
        .materialize(&ArtifactMaterializationRequest {
            run_id: ctx.run_id.clone(),
            bundle_id: crate::ids::BundleId::new("bundle:refit.capture").unwrap(),
            node_id: artifact.node_id.clone(),
            phase: Phase::Predict,
            variant_id: ctx.variant_id.clone(),
            controller_id: artifact.controller_id.clone(),
            artifact: artifact.artifact.clone(),
            params_fingerprint: artifact.params_fingerprint.clone(),
        })
        .unwrap();
    assert_eq!(
        handle,
        HandleRef {
            handle: 10_022,
            kind: HandleKind::Model,
            owner_controller: ControllerId::new("controller:model.mock").unwrap(),
        }
    );
}

#[test]
fn parallel_campaign_refit_captures_emitted_artifact_handles() {
    let plan = fixture_plan("plan:parallel.refit.artifact.capture");
    let provider = replay_data_provider();
    let mut artifact_store = InMemoryArtifactStore::new();
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(ReplayMockController {
            id: ControllerId::new("controller:transform.mock").unwrap(),
            handle: 11,
            require_artifact: false,
            emit_prediction: false,
            emit_refit_artifact: false,
        }))
        .unwrap();
    controllers
        .register(Box::new(ReplayMockController {
            id: ControllerId::new("controller:model.mock").unwrap(),
            handle: 22,
            require_artifact: false,
            emit_prediction: true,
            emit_refit_artifact: true,
        }))
        .unwrap();
    let mut ctx = RunContext::new(
        RunId::new("run:parallel.refit.artifact.capture").unwrap(),
        Some(11),
    );
    ctx.variant_id = Some(plan.variants[0].variant_id.clone());

    let results = ParallelScheduler::new(2)
        .unwrap()
        .execute_campaign_phase_with_data_provider_and_artifact_store(
            &plan,
            &controllers,
            &provider,
            &mut artifact_store,
            &mut ctx,
            Phase::Refit,
        )
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(artifact_store.len(), 1);
    assert_eq!(
        artifact_store.refit_artifacts()[0].artifact.id.as_str(),
        "artifact:model:base:refit"
    );
}

#[test]
fn node_result_validation_rejects_external_conformance_mismatches() {
    let plan = build_execution_plan(
        "plan:result.validation",
        simple_graph(),
        CampaignSpec {
            id: "campaign:result.validation".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let node_plan = plan
        .node_plans
        .get(&NodeId::new("model:pls").unwrap())
        .unwrap()
        .clone();
    let task = NodeTask {
        run_id: RunId::new("run:result.validation").unwrap(),
        node_plan: node_plan.clone(),
        phase: Phase::FitCv,
        variant_id: None,
        variant: None,
        fold_id: None,
        branch_path: Vec::new(),
        input_handles: BTreeMap::new(),
        data_views: BTreeMap::new(),
        prediction_inputs: BTreeMap::new(),
        artifact_inputs: BTreeMap::new(),
        seed: Some(99),
    };
    let controller = MockController {
        id: node_plan.controller_id.clone(),
        handle: 2,
        emit_prediction: false,
    };
    let result = controller.invoke(&task).unwrap();
    result.validate_for_task(&task).unwrap();

    let mut bad_controller = result.clone();
    bad_controller.lineage.controller_id = ControllerId::new("controller:wrong").unwrap();
    assert!(bad_controller
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("controller"));

    let mut bad_params = result.clone();
    bad_params.lineage.params_fingerprint = "wrong".to_string();
    assert!(bad_params
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("params fingerprint"));

    let mut bad_output_owner = result.clone();
    bad_output_owner
        .outputs
        .get_mut("out")
        .unwrap()
        .owner_controller = ControllerId::new("controller:wrong").unwrap();
    assert!(bad_output_owner
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("output `out`"));
}

#[test]
fn node_result_validation_checks_shape_fingerprints_and_feature_deltas() {
    let model_id = NodeId::new("model:pls").unwrap();
    let initial_feature_schema = "a".repeat(64);
    let updated_feature_schema = "b".repeat(64);
    let shape_plan = DataModelShapePlan {
        node_id: model_id.clone(),
        input_granularity: Granularity::Sample,
        target_granularity: Granularity::Sample,
        fit_rows: FitBoundary::FoldTrain,
        predict_rows: FitBoundary::FoldValidation,
        feature_namespace: Some("raw.x".to_string()),
        feature_schema_fingerprint: Some(initial_feature_schema.clone()),
        target_space: "raw".to_string(),
        aggregation_policy: AggregationPolicy::default(),
        augmentation_policy: Default::default(),
        selection_policy: Default::default(),
    };
    let plan = build_execution_plan(
        "plan:result.validation.shape",
        simple_graph(),
        CampaignSpec {
            id: "campaign:result.validation.shape".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::from([(model_id.clone(), shape_plan.clone())]),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let node_plan = plan.node_plans.get(&model_id).unwrap().clone();
    let task = NodeTask {
        run_id: RunId::new("run:result.validation.shape").unwrap(),
        node_plan: node_plan.clone(),
        phase: Phase::FitCv,
        variant_id: None,
        variant: None,
        fold_id: None,
        branch_path: Vec::new(),
        input_handles: BTreeMap::new(),
        data_views: BTreeMap::new(),
        prediction_inputs: BTreeMap::new(),
        artifact_inputs: BTreeMap::new(),
        seed: Some(99),
    };
    let controller = MockController {
        id: node_plan.controller_id.clone(),
        handle: 2,
        emit_prediction: false,
    };
    let mut result = controller.invoke(&task).unwrap();
    result.lineage.data_model_shape_fingerprint =
        Some(stable_json_fingerprint(&shape_plan).unwrap());
    result.lineage.aggregation_policy_fingerprint =
        Some(stable_json_fingerprint(&shape_plan.aggregation_policy).unwrap());
    result.shape_deltas = vec![ShapeDelta {
        node_id: model_id.clone(),
        kind: ShapeDeltaKind::Feature,
        before_fingerprint: initial_feature_schema.clone(),
        after_fingerprint: updated_feature_schema.clone(),
        metadata: BTreeMap::from([(
            "feature_namespace".to_string(),
            serde_json::Value::String("selected.x".to_string()),
        )]),
    }];
    result.validate_for_task(&task).unwrap();

    let mut wrong_shape_fingerprint = result.clone();
    wrong_shape_fingerprint.lineage.data_model_shape_fingerprint = Some("0".repeat(64));
    assert!(wrong_shape_fingerprint
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("data/model shape fingerprint"));

    let mut wrong_feature_delta = result.clone();
    wrong_feature_delta.shape_deltas[0].before_fingerprint = "c".repeat(64);
    assert!(wrong_feature_delta
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("expected current schema"));

    let mut unchanged_delta = result;
    unchanged_delta.shape_deltas[0].after_fingerprint = initial_feature_schema;
    assert!(unchanged_delta
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("does not change fingerprint"));
}

#[test]
fn node_result_validation_rejects_bad_artifact_handles() {
    let plan = build_execution_plan(
        "plan:result.validation.artifacts",
        simple_graph(),
        CampaignSpec {
            id: "campaign:result.validation.artifacts".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: None,
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::new(),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let node_plan = plan
        .node_plans
        .get(&NodeId::new("model:pls").unwrap())
        .unwrap()
        .clone();
    let task = NodeTask {
        run_id: RunId::new("run:result.validation.artifacts").unwrap(),
        node_plan: node_plan.clone(),
        phase: Phase::Refit,
        variant_id: None,
        variant: None,
        fold_id: None,
        branch_path: Vec::new(),
        input_handles: BTreeMap::new(),
        data_views: BTreeMap::new(),
        prediction_inputs: BTreeMap::new(),
        artifact_inputs: BTreeMap::new(),
        seed: Some(99),
    };
    let controller = MockController {
        id: node_plan.controller_id.clone(),
        handle: 2,
        emit_prediction: false,
    };
    let base = controller.invoke(&task).unwrap();
    let artifact = ArtifactRef {
        id: ArtifactId::new("artifact:model:pls:refit").unwrap(),
        kind: "mock_model".to_string(),
        controller_id: node_plan.controller_id.clone(),
        backend: None,
        uri: None,
        content_fingerprint: None,
        size_bytes: Some(128),
        plugin: None,
        plugin_version: None,
    };
    let handle = HandleRef {
        handle: 77,
        kind: HandleKind::Model,
        owner_controller: node_plan.controller_id.clone(),
    };
    let mut valid = base.clone();
    valid.artifacts = vec![artifact.clone()];
    valid
        .artifact_handles
        .insert(artifact.id.clone(), handle.clone());
    valid.lineage.artifact_refs = vec![artifact.clone()];
    valid.validate_for_task(&task).unwrap();

    let mut missing_handle = valid.clone();
    missing_handle.artifact_handles.clear();
    assert!(missing_handle
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("without artifact handle"));

    let mut wrong_kind = valid.clone();
    wrong_kind
        .artifact_handles
        .get_mut(&artifact.id)
        .unwrap()
        .kind = HandleKind::Data;
    assert!(wrong_kind
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("non-artifact/model handle kind"));

    let mut wrong_owner = valid.clone();
    wrong_owner
        .artifact_handles
        .get_mut(&artifact.id)
        .unwrap()
        .owner_controller = ControllerId::new("controller:wrong").unwrap();
    assert!(wrong_owner
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("owned by"));

    let mut undeclared_handle = base.clone();
    undeclared_handle.artifact_handles.insert(
        ArtifactId::new("artifact:model:pls:extra").unwrap(),
        handle.clone(),
    );
    assert!(undeclared_handle
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("undeclared artifact"));

    let mut missing_lineage_ref = valid;
    missing_lineage_ref.lineage.artifact_refs.clear();
    assert!(missing_lineage_ref
        .validate_for_task(&task)
        .unwrap_err()
        .to_string()
        .contains("lineage artifact ref"));
}

#[test]
fn artifact_ref_validates_portable_metadata() {
    let content_fingerprint = "a".repeat(64);
    let artifact = ArtifactRef {
        id: ArtifactId::new("artifact:model:portable").unwrap(),
        kind: "model".to_string(),
        controller_id: ControllerId::new("controller:sklearn").unwrap(),
        backend: Some(ArtifactBackend::Joblib),
        uri: Some(format!("artifacts/{content_fingerprint}.joblib")),
        content_fingerprint: Some(content_fingerprint.clone()),
        size_bytes: Some(4096),
        plugin: Some("dagml.sklearn".to_string()),
        plugin_version: Some("1.0.0".to_string()),
    };

    artifact.validate().unwrap();
    let encoded = serde_json::to_value(&artifact).unwrap();
    assert_eq!(encoded["backend"].as_str(), Some("joblib"));
    assert_eq!(
        encoded["content_fingerprint"].as_str(),
        Some(content_fingerprint.as_str())
    );

    let legacy: ArtifactRef = serde_json::from_value(serde_json::json!({
        "id": "artifact:model:legacy",
        "kind": "mock_model",
        "controller_id": "controller:mock",
        "size_bytes": 128
    }))
    .unwrap();
    assert_eq!(legacy.backend, None);
    assert_eq!(legacy.content_fingerprint, None);
    legacy.validate().unwrap();
}

#[test]
fn artifact_ref_rejects_invalid_portable_metadata() {
    let mut artifact = ArtifactRef {
        id: ArtifactId::new("artifact:model:portable").unwrap(),
        kind: "model".to_string(),
        controller_id: ControllerId::new("controller:sklearn").unwrap(),
        backend: Some(ArtifactBackend::Joblib),
        uri: Some("artifacts/model.joblib".to_string()),
        content_fingerprint: Some("b".repeat(64)),
        size_bytes: Some(4096),
        plugin: Some("dagml.sklearn".to_string()),
        plugin_version: Some("1.0.0".to_string()),
    };
    artifact.validate().unwrap();

    let mut bad_fingerprint = artifact.clone();
    bad_fingerprint.content_fingerprint = Some("not-a-digest".to_string());
    assert!(bad_fingerprint
        .validate()
        .unwrap_err()
        .to_string()
        .contains("artifact content fingerprint"));

    let mut missing_backend = artifact.clone();
    missing_backend.backend = None;
    assert!(missing_backend
        .validate()
        .unwrap_err()
        .to_string()
        .contains("uri without backend"));

    let mut missing_fingerprint = artifact.clone();
    missing_fingerprint.content_fingerprint = None;
    assert!(missing_fingerprint
        .validate()
        .unwrap_err()
        .to_string()
        .contains("uri without content_fingerprint"));

    artifact.plugin = None;
    assert!(artifact
        .validate()
        .unwrap_err()
        .to_string()
        .contains("plugin_version without plugin"));
}

#[test]
fn node_result_validation_rejects_predictions_outside_validation_view() {
    let model_id = NodeId::new("model:pls").unwrap();
    let plan = build_execution_plan(
        "plan:result.validation.samples",
        simple_graph(),
        CampaignSpec {
            id: "campaign:result.validation.samples".to_string(),
            root_seed: Some(11),
            leakage_policy: Default::default(),
            aggregation_policy: Default::default(),
            split_invocation: Some(SplitInvocation {
                id: "split:outer".to_string(),
                controller_id: None,
                leakage_policy: Default::default(),
                params: BTreeMap::new(),
                fold_set: Some(two_fold_set()),
            }),
            generation: Default::default(),
            shape_plans: BTreeMap::new(),
            data_bindings: BTreeMap::from([(model_id.clone(), vec![data_binding(&model_id)])]),
            branch_view_plans: Vec::new(),
            metadata: BTreeMap::new(),
        },
        &manifests(),
    )
    .unwrap();
    let node_plan = plan.node_plans.get(&model_id).unwrap().clone();
    let task = NodeTask {
        run_id: RunId::new("run:result.validation.samples").unwrap(),
        node_plan: node_plan.clone(),
        phase: Phase::FitCv,
        variant_id: Some(VariantId::new("variant:base").unwrap()),
        variant: None,
        fold_id: Some(FoldId::new("fold:0").unwrap()),
        branch_path: Vec::new(),
        input_handles: BTreeMap::new(),
        data_views: BTreeMap::from([(
            "data:x:validation".to_string(),
            DataProviderViewSpec {
                sample_ids: Some(vec![SampleId::new("s1").unwrap()]),
                partition: DataRequestPartition::FoldValidation,
                fold_id: Some(FoldId::new("fold:0").unwrap()),
                source_ids: None,
                columns: None,
                include_augmented: false,
                include_excluded: false,
                branch_view: None,
                extra: BTreeMap::new(),
            },
        )]),
        prediction_inputs: BTreeMap::new(),
        artifact_inputs: BTreeMap::new(),
        seed: Some(99),
    };
    let result = NodeResult {
        node_id: model_id.clone(),
        outputs: BTreeMap::from([(
            "out".to_string(),
            HandleRef {
                handle: 7,
                kind: HandleKind::Data,
                owner_controller: node_plan.controller_id.clone(),
            },
        )]),
        predictions: vec![PredictionBlock {
            prediction_id: Some("pred:bad.sample".to_string()),
            producer_node: model_id,
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            sample_ids: vec![SampleId::new("s2").unwrap()],
            values: vec![vec![1.0]],
            target_names: vec!["y".to_string()],
        }],
        observation_predictions: Vec::new(),
        aggregated_predictions: Vec::new(),
        shape_deltas: Vec::new(),
        artifacts: Vec::new(),
        artifact_handles: BTreeMap::new(),
        lineage: LineageRecord {
            record_id: LineageId::new("lineage:bad.sample").unwrap(),
            run_id: task.run_id.clone(),
            node_id: task.node_plan.node_id.clone(),
            phase: task.phase,
            controller_id: task.node_plan.controller_id.clone(),
            controller_version: task.node_plan.controller_version.clone(),
            variant_id: task.variant_id.clone(),
            fold_id: task.fold_id.clone(),
            branch_path: task.branch_path.clone(),
            input_lineage: Vec::new(),
            artifact_refs: Vec::new(),
            params_fingerprint: task.node_plan.params_fingerprint.clone(),
            data_model_shape_fingerprint: None,
            aggregation_policy_fingerprint: None,
            seed: task.seed,
            unsafe_flags: BTreeSet::new(),
            metrics: BTreeMap::new(),
        },
    };

    assert!(result.validate_for_task(&task).is_err());
}

#[test]
fn in_memory_artifact_store_resolves_bundle_artifacts() {
    let plan = fixture_plan("plan:replay.artifacts");
    let bundle = replay_bundle(&plan);
    let artifact = &bundle.refit_artifacts[0];
    let mut store = InMemoryArtifactStore::new();
    let handle = HandleRef {
        handle: 77,
        kind: HandleKind::Model,
        owner_controller: artifact.controller_id.clone(),
    };
    store.register(artifact, handle.clone()).unwrap();

    let resolved = store
        .materialize(&ArtifactMaterializationRequest {
            run_id: RunId::new("run:replay.artifacts").unwrap(),
            bundle_id: bundle.bundle_id.clone(),
            node_id: artifact.node_id.clone(),
            phase: Phase::Predict,
            variant_id: bundle.selected_variant_id.clone(),
            controller_id: artifact.controller_id.clone(),
            artifact: artifact.artifact.clone(),
            params_fingerprint: artifact.params_fingerprint.clone(),
        })
        .unwrap();

    assert_eq!(resolved, handle);
    assert_eq!(store.len(), 1);
    assert!(InMemoryArtifactStore::new()
        .materialize(&ArtifactMaterializationRequest {
            run_id: RunId::new("run:replay.artifacts").unwrap(),
            bundle_id: bundle.bundle_id.clone(),
            node_id: artifact.node_id.clone(),
            phase: Phase::Predict,
            variant_id: bundle.selected_variant_id.clone(),
            controller_id: artifact.controller_id.clone(),
            artifact: artifact.artifact.clone(),
            params_fingerprint: artifact.params_fingerprint.clone(),
        })
        .is_err());
}

#[test]
fn bundle_replay_invokes_predict_with_data_and_refit_artifact_handles() {
    let plan = fixture_plan("plan:replay.predict");
    let bundle = replay_bundle(&plan);
    let request = replay_request(&bundle, Phase::Predict);
    let envelopes = replay_envelopes();
    let provider = replay_data_provider();
    let store = replay_artifact_store(&bundle);
    let controllers = replay_runtime_controllers();
    let mut ctx = RunContext::new(RunId::new("run:replay.predict").unwrap(), Some(11));

    let results = SequentialScheduler
        .execute_bundle_replay(
            BundleReplayExecution {
                plan: &plan,
                bundle: &bundle,
                replay_request: &request,
                prediction_cache_store: None,
                controllers: &controllers,
                data_provider: &provider,
                artifact_store: &store,
                data_envelopes: &envelopes,
            },
            &mut ctx,
        )
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(provider.handle_records().len(), 1);
    assert_eq!(provider.view_records().len(), 1);
    assert_eq!(
        provider.view_records()[0].view.partition,
        DataRequestPartition::Predict
    );
    assert_eq!(ctx.prediction_store.blocks().len(), 1);
    assert_eq!(
        ctx.prediction_store.blocks()[0].partition,
        PredictionPartition::Final
    );
    assert!(ctx
        .lineage
        .records()
        .any(|record| record.node_id.as_str() == "model:base"
            && record.phase == Phase::Predict
            && record.variant_id == bundle.selected_variant_id));

    let provider = replay_data_provider();
    let mut ctx = RunContext::new(RunId::new("run:parallel.replay.predict").unwrap(), Some(11));
    let results = ParallelScheduler::new(2)
        .unwrap()
        .execute_bundle_replay(
            BundleReplayExecution {
                plan: &plan,
                bundle: &bundle,
                replay_request: &request,
                prediction_cache_store: None,
                controllers: &controllers,
                data_provider: &provider,
                artifact_store: &store,
                data_envelopes: &envelopes,
            },
            &mut ctx,
        )
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(provider.handle_records().len(), 1);
    assert_eq!(provider.view_records().len(), 1);
    assert_eq!(
        provider.view_records()[0].view.partition,
        DataRequestPartition::Predict
    );
    assert_eq!(ctx.prediction_store.blocks().len(), 1);
}

#[test]
fn bundle_replay_rejects_missing_artifact_unsupported_phase_and_bad_envelope() {
    let plan = fixture_plan("plan:replay.reject");
    let bundle = replay_bundle(&plan);
    let request = replay_request(&bundle, Phase::Predict);
    let envelopes = replay_envelopes();
    let provider = replay_data_provider();
    let controllers = replay_runtime_controllers();
    let mut ctx = RunContext::new(RunId::new("run:replay.reject").unwrap(), Some(11));

    assert!(SequentialScheduler
        .execute_bundle_replay(
            BundleReplayExecution {
                plan: &plan,
                bundle: &bundle,
                replay_request: &request,
                prediction_cache_store: None,
                controllers: &controllers,
                data_provider: &provider,
                artifact_store: &InMemoryArtifactStore::new(),
                data_envelopes: &envelopes,
            },
            &mut ctx,
        )
        .is_err());

    let store = replay_artifact_store(&bundle);
    assert!(SequentialScheduler
        .execute_bundle_replay(
            BundleReplayExecution {
                plan: &plan,
                bundle: &bundle,
                replay_request: &replay_request(&bundle, Phase::FitCv),
                prediction_cache_store: None,
                controllers: &controllers,
                data_provider: &provider,
                artifact_store: &store,
                data_envelopes: &envelopes,
            },
            &mut ctx,
        )
        .is_err());

    let mut bad_envelopes = replay_envelopes();
    bad_envelopes
        .get_mut("model:base.x")
        .unwrap()
        .schema_fingerprint = "0".repeat(64);
    assert!(SequentialScheduler
        .execute_bundle_replay(
            BundleReplayExecution {
                plan: &plan,
                bundle: &bundle,
                replay_request: &request,
                prediction_cache_store: None,
                controllers: &controllers,
                data_provider: &provider,
                artifact_store: &store,
                data_envelopes: &bad_envelopes,
            },
            &mut ctx,
        )
        .is_err());
}
