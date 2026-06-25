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
use crate::fold::{FoldAssignment, FoldPartitionMode, FoldSet};
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
    FitBoundary, FitInfluencePolicy, Granularity, LeakageUnitPolicy, ShapeDelta, ShapeDeltaKind,
    SplitUnit,
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
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
            explanations: Vec::new(),
            shape_deltas: vec![shape_delta],
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: artifacts.clone(),
            artifact_handles,
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
                    reduction_plan: None,
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
                    reduction_plan: None,
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
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
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
        unit_level: None,
        alignment_key: None,
        target_level: None,
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
        inner_cv: None,
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
        inner_cv: None,
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
                partition_mode: FoldPartitionMode::Partition,
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
                requires_oof: true,
                requires_fold_alignment: true,
                ..EdgeContract::new(PortKind::Prediction, None)
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
        inner_cv: None,
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
                partition_mode: FoldPartitionMode::Partition,
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
                requires_oof: false,
                requires_fold_alignment: false,
                ..EdgeContract::new(PortKind::Data, None)
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
                requires_oof: false,
                requires_fold_alignment: false,
                ..EdgeContract::new(PortKind::Data, None)
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
                requires_oof: false,
                requires_fold_alignment: true,
                ..EdgeContract::new(PortKind::Prediction, None)
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
                requires_oof: true,
                requires_fold_alignment: true,
                ..EdgeContract::new(PortKind::Prediction, None)
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
        partition_mode: FoldPartitionMode::Partition,
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
        partition_mode: FoldPartitionMode::Partition,
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
        partition_mode: FoldPartitionMode::Partition,
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
    let mut relation = SampleRelation::new(
        ObservationId::new(observation_id).unwrap(),
        SampleId::new(sample_id).unwrap(),
    );
    relation.target_id = Some(TargetId::new(target_id).unwrap());
    relation.group_id = Some(GroupId::new(group_id).unwrap());
    relation.origin_sample_id = origin_sample_id.map(|value| SampleId::new(value).unwrap());
    relation.source_id = Some("nir".to_string());
    relation.is_augmented = is_augmented;
    relation
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
        inner_cv: None,
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
        inner_cv: None,
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
        inner_cv: None,
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
            inner_cv: None,
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
                explanations: Vec::new(),
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                fit_influence_diagnostics: Vec::new(),
                regression_targets: Vec::new(),
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
            inner_cv: None,
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
                explanations: Vec::new(),
                shape_deltas: Vec::new(),
                artifacts: Vec::new(),
                artifact_handles: BTreeMap::new(),
                fit_influence_diagnostics: Vec::new(),
                regression_targets: Vec::new(),
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
            inner_cv: None,
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
            inner_cv: None,
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
fn d9_golden_oof_refit_and_predict_replay_mock_run() {
    #[derive(serde::Deserialize)]
    struct D9GoldenFixture {
        golden_scenarios: Vec<D9GoldenScenario>,
    }

    #[derive(serde::Deserialize)]
    struct D9GoldenScenario {
        scenario_id: String,
        mock_phase_path: Vec<String>,
    }

    let fixture: D9GoldenFixture = serde_json::from_str(include_str!(
        "../../../../examples/fixtures/runtime/d9_golden_multisource_scenarios.json"
    ))
    .unwrap();
    assert_eq!(fixture.golden_scenarios.len(), 7);

    for (index, scenario) in fixture.golden_scenarios.iter().enumerate() {
        assert_eq!(scenario.mock_phase_path, ["fit_cv", "refit", "predict"]);
        let oof_plan = build_execution_plan(
            format!("plan:d9.oof.refit.{index}"),
            oof_edge_graph(),
            oof_edge_campaign(),
            &oof_edge_manifests(BTreeSet::from([Phase::FitCv, Phase::Refit])),
        )
        .unwrap();
        let mut oof_ctx = RunContext::new(
            RunId::new(format!("run:d9.oof.refit.{index}")).unwrap(),
            Some(11),
        );
        let fit_controllers = oof_edge_runtime_controllers(
            Some(PredictionPartition::Validation),
            OofSampleMode::Aligned,
        );
        let fit_results = SequentialScheduler
            .execute_campaign_phase(&oof_plan, &fit_controllers, &mut oof_ctx, Phase::FitCv)
            .unwrap();
        assert_eq!(
            fit_results.len(),
            4,
            "{} did not mock-run fit_cv through OOF",
            scenario.scenario_id
        );
        assert_eq!(
            oof_ctx.prediction_store.blocks().len(),
            2,
            "{} did not emit complete validation OOF",
            scenario.scenario_id
        );

        let refit_controllers = oof_edge_runtime_controllers(None, OofSampleMode::Aligned);
        let refit_results = SequentialScheduler
            .execute_campaign_phase(&oof_plan, &refit_controllers, &mut oof_ctx, Phase::Refit)
            .unwrap();
        assert_eq!(
            refit_results
                .iter()
                .filter(|result| result.node_id.as_str() == "model:meta")
                .count(),
            1,
            "{} did not mock-run refit with full OOF coverage",
            scenario.scenario_id
        );

        let replay_plan = fixture_plan(&format!("plan:d9.predict.replay.{index}"));
        let bundle = replay_bundle(&replay_plan);
        let request = replay_request(&bundle, Phase::Predict);
        let envelopes = replay_envelopes();
        let provider = replay_data_provider();
        let store = replay_artifact_store(&bundle);
        let controllers = replay_runtime_controllers();
        let mut replay_ctx = RunContext::new(
            RunId::new(format!("run:d9.predict.replay.{index}")).unwrap(),
            Some(11),
        );
        let replay_results = SequentialScheduler
            .execute_bundle_replay(
                BundleReplayExecution {
                    plan: &replay_plan,
                    bundle: &bundle,
                    replay_request: &request,
                    prediction_cache_store: None,
                    controllers: &controllers,
                    data_provider: &provider,
                    artifact_store: &store,
                    data_envelopes: &envelopes,
                },
                &mut replay_ctx,
            )
            .unwrap();

        assert_eq!(
            replay_results.len(),
            2,
            "{} did not mock-run predict replay",
            scenario.scenario_id
        );
        assert_eq!(provider.view_records().len(), 1);
        assert_eq!(
            provider.view_records()[0].view.partition,
            DataRequestPartition::Predict
        );
        assert_eq!(replay_ctx.prediction_store.blocks().len(), 1);
        assert_eq!(
            replay_ctx.prediction_store.blocks()[0].partition,
            PredictionPartition::Final
        );
    }
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
            inner_cv: None,
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
            inner_cv: None,
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
                requires_oof: false,
                requires_fold_alignment: false,
                ..EdgeContract::new(PortKind::Data, None)
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
            inner_cv: None,
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
        representation_plan: None,
        representation_replay_manifest: None,
        representation_compatibility: None,
        relation_delta_fingerprint: None,
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
            topological_order: vec![node_id.clone(), NodeId::new("model:plain").unwrap()],
            parallel_levels: Vec::new(),
        },
        campaign: crate::plan::CampaignSpec {
            inner_cv: None,
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
        super::branch_view_from_node_metadata(&plan, &NodeId::new("model:plain").unwrap()).unwrap();
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
    let properties = schema["properties"].as_object().unwrap();
    assert!(properties.contains_key("representation_plan"));
    assert!(properties.contains_key("representation_replay_manifest"));
    assert!(properties.contains_key("representation_compatibility"));
    assert!(properties.contains_key("relation_delta_fingerprint"));
    let defs = schema["$defs"].as_object().unwrap();
    assert!(defs.contains_key("combination_plan"));
    assert!(defs.contains_key("representation_plan"));
    assert!(defs.contains_key("representation_replay_manifest"));
    assert!(defs.contains_key("representation_compatibility_report"));
    assert!(defs.contains_key("representation_sample_observation_mapping"));
    assert!(defs.contains_key("representation_combo_selection_record"));
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

fn fit_influence_view(sample_ids: Vec<&str>) -> BTreeMap<String, DataProviderViewSpec> {
    BTreeMap::from([(
        "data:x:train".to_string(),
        DataProviderViewSpec {
            sample_ids: Some(
                sample_ids
                    .into_iter()
                    .map(|sample_id| SampleId::new(sample_id).unwrap())
                    .collect(),
            ),
            partition: DataRequestPartition::FoldTrain,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            source_ids: None,
            columns: None,
            include_augmented: false,
            include_excluded: false,
            branch_view: None,
            extra: BTreeMap::new(),
        },
    )])
}

#[test]
fn fit_influence_strict_requires_weight_support() {
    let error = resolve_fit_influence_task(
        FitInfluencePolicy::StrictWeightSupport,
        &BTreeSet::new(),
        &fit_influence_view(vec!["s1", "s1", "s2"]),
    )
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("fit influence"),
        "unexpected strict support error: {error}"
    );
}

#[test]
fn d9_negative_controller_lacking_fit_influence_capability_is_rejected() {
    let error = resolve_fit_influence_task(
        FitInfluencePolicy::EqualSampleInfluence,
        &BTreeSet::new(),
        &fit_influence_view(vec!["s1", "s1", "s2"]),
    )
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("controller capabilities do not support requested fit influence policy"),
        "unexpected D9 fit-influence capability error: {error}"
    );
}

#[test]
fn fit_influence_auto_falls_back_with_warning() {
    let task = resolve_fit_influence_task(
        FitInfluencePolicy::Auto,
        &BTreeSet::new(),
        &fit_influence_view(vec!["s1", "s1", "s2"]),
    )
    .unwrap();

    assert_eq!(task.effective_policy, FitInfluencePolicy::UniformRows);
    assert_eq!(task.mechanism, FitInfluenceMechanism::UniformRows);
    assert!(task.warnings[0].contains("fell back"));
    task.validate().unwrap();

    let diagnostic = task.diagnostic();
    assert!(diagnostic.fallback_used);
    assert_eq!(diagnostic.row_weight_count, 0);
    assert_eq!(diagnostic.warnings, task.warnings);
}

#[test]
fn equal_sample_influence_emits_per_row_weights_without_aggregation_weights() {
    let capabilities = BTreeSet::from([ControllerCapability::SupportsSampleWeights]);
    let task = resolve_fit_influence_task(
        FitInfluencePolicy::EqualSampleInfluence,
        &capabilities,
        &fit_influence_view(vec!["s1", "s1", "s2"]),
    )
    .unwrap();

    assert_eq!(task.mechanism, FitInfluenceMechanism::SampleWeights);
    assert_eq!(task.row_weights, vec![0.5, 0.5, 1.0]);

    let aggregation = AggregationPolicy {
        method: AggregationMethod::WeightedMean,
        weights: crate::policy::AggregationWeights::RepetitionCount,
        ..AggregationPolicy::default()
    };
    aggregation.validate().unwrap();
    assert_eq!(
        task.effective_policy,
        FitInfluencePolicy::EqualSampleInfluence
    );
}

#[test]
fn node_result_validation_rejects_external_conformance_mismatches() {
    let plan = build_execution_plan(
        "plan:result.validation",
        simple_graph(),
        CampaignSpec {
            inner_cv: None,
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
        inner_fold_set: None,
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
        fit_influence: FitInfluenceTask::default(),
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
            inner_cv: None,
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
        inner_fold_set: None,
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
        fit_influence: FitInfluenceTask::default(),
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
            inner_cv: None,
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
        inner_fold_set: None,
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
        fit_influence: FitInfluenceTask::default(),
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
            inner_cv: None,
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
        inner_fold_set: None,
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
        fit_influence: FitInfluenceTask::default(),
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
        explanations: Vec::new(),
        shape_deltas: Vec::new(),
        artifacts: Vec::new(),
        artifact_handles: BTreeMap::new(),
        fit_influence_diagnostics: Vec::new(),
        regression_targets: Vec::new(),
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

#[test]
fn fit_cv_node_with_inner_cv_carries_inner_fold_set_subset_of_outer_train() {
    use crate::fold::{KFoldSpec, NestedCvSpec};
    use crate::ids::SampleId;

    // Reuse a real plan's campaign + a node plan, but drive nesting with a fresh
    // outer fold set that has enough train samples for an inner KFold.
    let plan = live_group_oof_runtime_plan();
    let mut campaign = plan.campaign.clone();
    let node_plan = plan
        .node_plans
        .values()
        .next()
        .expect("plan has at least one node")
        .clone();
    assert!(
        node_plan.inner_cv.is_none(),
        "node falls back to campaign default"
    );

    let samples = ["s1", "s2", "s3", "s4"]
        .into_iter()
        .map(|s| SampleId::new(s).unwrap())
        .collect::<Vec<_>>();
    let outer = KFoldSpec {
        n_splits: 2,
        shuffle: false,
        seed: Some(0),
    }
    .split("outer", &samples)
    .unwrap();
    let outer_fold = outer.folds[0].clone();
    let fit_scope = PhaseScope {
        phase: Phase::FitCv,
        variant_id: None,
        variant: None,
        fold_id: Some(outer_fold.fold_id.clone()),
        seed_root: None,
    };

    // With a campaign-level inner CV and no node override, the node gets an inner
    // fold set built from this outer fold's TRAIN samples (⊆ outer-train).
    campaign.inner_cv = Some(NestedCvSpec::KFold(KFoldSpec {
        n_splits: 2,
        shuffle: false,
        seed: Some(1),
    }));
    let inner = inner_fold_set_for_scope(&campaign, Some(&outer), &node_plan, &fit_scope)
        .expect("inner fold set builds")
        .expect("inner fold set present for FIT_CV node with inner_cv");
    let outer_train = outer_fold
        .train_sample_ids
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    for sample_id in &inner.sample_ids {
        assert!(
            outer_train.contains(sample_id),
            "inner sample escapes outer-train"
        );
    }
    assert_eq!(
        inner
            .sample_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>(),
        outer_train
    );

    // No effective inner CV → no inner fold set.
    campaign.inner_cv = None;
    assert!(
        inner_fold_set_for_scope(&campaign, Some(&outer), &node_plan, &fit_scope)
            .unwrap()
            .is_none()
    );

    // Non-FIT_CV phases never carry an inner fold set, even with inner_cv declared.
    campaign.inner_cv = Some(NestedCvSpec::KFold(KFoldSpec {
        n_splits: 2,
        shuffle: false,
        seed: Some(1),
    }));
    let predict_scope = PhaseScope {
        phase: Phase::Predict,
        variant_id: None,
        variant: None,
        fold_id: None,
        seed_root: None,
    };
    assert!(
        inner_fold_set_for_scope(&campaign, Some(&outer), &node_plan, &predict_scope)
            .unwrap()
            .is_none()
    );
}

#[test]
fn native_scoring_collects_reports_and_builds_score_set() {
    use crate::aggregation::PredictionUnitId;
    use crate::ids::SampleId;
    use crate::metrics::RegressionTargetBlock;
    use crate::policy::PredictionLevel;

    let node = NodeId::new("model:pls").unwrap();
    let predictions = PredictionBlock {
        prediction_id: None,
        producer_node: node.clone(),
        partition: PredictionPartition::Validation,
        fold_id: None,
        sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
        values: vec![vec![2.0], vec![4.0]],
        target_names: vec!["y".to_string()],
    };
    let targets = RegressionTargetBlock {
        level: PredictionLevel::Sample,
        unit_ids: vec![
            PredictionUnitId::Sample(SampleId::new("s1").unwrap()),
            PredictionUnitId::Sample(SampleId::new("s2").unwrap()),
        ],
        values: vec![vec![2.0], vec![4.0]],
        target_names: vec!["y".to_string()],
    };
    let make = |regression_targets: Vec<RegressionTargetBlock>| NodeResult {
        node_id: node.clone(),
        outputs: BTreeMap::new(),
        predictions: vec![predictions.clone()],
        observation_predictions: Vec::new(),
        aggregated_predictions: Vec::new(),
        explanations: Vec::new(),
        shape_deltas: Vec::new(),
        artifacts: Vec::new(),
        artifact_handles: BTreeMap::new(),
        fit_influence_diagnostics: Vec::new(),
        regression_targets,
        lineage: LineageRecord {
            record_id: LineageId::new("lineage:t").unwrap(),
            run_id: RunId::new("run:t").unwrap(),
            node_id: node.clone(),
            phase: Phase::FitCv,
            controller_id: ControllerId::new("controller:pls").unwrap(),
            controller_version: "1".to_string(),
            variant_id: None,
            fold_id: None,
            branch_path: Vec::new(),
            input_lineage: Vec::new(),
            artifact_refs: Vec::new(),
            params_fingerprint: "fp".to_string(),
            data_model_shape_fingerprint: None,
            aggregation_policy_fingerprint: None,
            seed: None,
            unsafe_flags: BTreeSet::new(),
            metrics: BTreeMap::new(),
        },
    };

    // Targets present -> the result is scored natively and collectable into a ScoreSet.
    let mut ctx = RunContext::new(RunId::new("run:t").unwrap(), None);
    apply_result_scoring(
        &make(vec![targets]),
        &mut ctx.score_collector,
        &mut ctx.regression_target_records,
    )
    .unwrap();
    assert_eq!(ctx.score_collector.len(), 1);
    assert_eq!(ctx.regression_target_records.len(), 1);
    assert!(ctx.score_collector[0].metrics.contains_key("rmse"));
    let set = ctx
        .build_score_set("plan:t", Some("rmse".to_string()))
        .unwrap();
    assert_eq!(set.reports.len(), 1);
    set.validate().unwrap();

    // No targets -> nothing collected, no ScoreSet (existing runs are unaffected).
    let mut empty = RunContext::new(RunId::new("run:t").unwrap(), None);
    apply_result_scoring(
        &make(Vec::new()),
        &mut empty.score_collector,
        &mut empty.regression_target_records,
    )
    .unwrap();
    assert!(empty.score_collector.is_empty());
    assert!(empty.build_score_set("plan:t", None).is_none());
}

#[test]
fn cross_fold_validation_reports_scores_the_oof_average() {
    use crate::aggregation::PredictionUnitId;
    use crate::ids::SampleId;
    use crate::metrics::{
        cross_fold_validation_reports, RegressionTargetBlock, RegressionTargetRecord,
    };
    use crate::policy::PredictionLevel;

    let node = NodeId::new("model:pls").unwrap();
    let pred = |fold: &str, rows: &[(&str, f64)]| PredictionBlock {
        prediction_id: None,
        producer_node: node.clone(),
        partition: PredictionPartition::Validation,
        fold_id: Some(FoldId::new(fold).unwrap()),
        sample_ids: rows
            .iter()
            .map(|(s, _)| SampleId::new(*s).unwrap())
            .collect(),
        values: rows.iter().map(|(_, v)| vec![*v]).collect(),
        target_names: vec!["y".to_string()],
    };
    let record = |fold: &str, rows: &[(&str, f64)]| RegressionTargetRecord {
        producer_node: node.clone(),
        variant_id: None,
        partition: PredictionPartition::Validation,
        fold_id: Some(FoldId::new(fold).unwrap()),
        block: RegressionTargetBlock {
            level: PredictionLevel::Sample,
            unit_ids: rows
                .iter()
                .map(|(s, _)| PredictionUnitId::Sample(SampleId::new(*s).unwrap()))
                .collect(),
            values: rows.iter().map(|(_, v)| vec![*v]).collect(),
            target_names: vec!["y".to_string()],
        },
    };

    // Two disjoint folds -> OOF concat scored over all 4 samples; residual only on s4 (5 vs 4).
    let blocks = [
        pred("fold0", &[("s1", 1.0), ("s2", 2.0)]),
        pred("fold1", &[("s3", 3.0), ("s4", 5.0)]),
    ];
    let records = [
        record("fold0", &[("s1", 1.0), ("s2", 2.0)]),
        record("fold1", &[("s3", 3.0), ("s4", 4.0)]),
    ];
    let reports = cross_fold_validation_reports(&blocks, &records, SCORE_METRICS).unwrap();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].fold_id, Some(FoldId::new("avg").unwrap()));
    assert_eq!(reports[0].partition, PredictionPartition::Validation);
    assert_eq!(reports[0].row_count, 4);
    assert!((reports[0].metrics["rmse"] - 0.5).abs() < 1e-9); // sqrt((0+0+0+1)/4)
}

/// Model controller for the native-variant-SELECT test. For each FIT_CV fold it emits one VALIDATION
/// prediction plus the matching `y_true`, keyed by `fold_id` (fold:0 -> s1, fold:1 -> s2). The
/// predicted value is `y_true + offset`, where `offset` is read from the variant's `n_components`
/// param override — so different variants yield different OOF residuals (hence different OOF RMSE).
struct VariantScoringController {
    id: ControllerId,
    handle: u64,
}

impl VariantScoringController {
    fn fold_sample(task: &NodeTask) -> Option<(SampleId, f64)> {
        // (validation sample, its y_true) per fold of `two_fold_set`.
        match task.fold_id.as_ref()?.as_str() {
            "fold:0" => Some((SampleId::new("s1").unwrap(), 1.0)),
            "fold:1" => Some((SampleId::new("s2").unwrap(), 2.0)),
            _ => None,
        }
    }
}

impl RuntimeController for VariantScoringController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        let output = HandleRef {
            handle: self.handle,
            kind: HandleKind::Data,
            owner_controller: self.id.clone(),
        };
        let mut predictions = Vec::new();
        let mut regression_targets = Vec::new();
        if let Some((sample_id, y_true)) = Self::fold_sample(task) {
            // Prediction = y_true + offset(variant). offset is the variant's `n_components` override.
            let offset = task
                .node_plan
                .params
                .get("n_components")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0);
            predictions.push(PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: PredictionPartition::Validation,
                fold_id: task.fold_id.clone(),
                sample_ids: vec![sample_id.clone()],
                values: vec![vec![y_true + offset]],
                target_names: vec!["y".to_string()],
            });
            regression_targets.push(crate::metrics::RegressionTargetBlock {
                level: PredictionLevel::Sample,
                unit_ids: vec![crate::aggregation::PredictionUnitId::Sample(sample_id)],
                values: vec![vec![y_true]],
                target_names: vec!["y".to_string()],
            });
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
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([("pred".to_string(), output)]),
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets,
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

fn variant_scoring_campaign(offsets: Vec<(&str, f64)>) -> CampaignSpec {
    let choices = offsets
        .into_iter()
        .map(|(label, offset)| GenerationChoice {
            label: label.to_string(),
            value: json!(label),
            param_overrides: vec![crate::generation::GenerationParamOverride {
                node_id: NodeId::new("model:pls").unwrap(),
                params: BTreeMap::from([("n_components".to_string(), json!(offset))]),
            }],
        })
        .collect::<Vec<_>>();
    let max_variants = Some(choices.len());
    CampaignSpec {
        inner_cv: None,
        id: "campaign:variant.select".to_string(),
        root_seed: Some(7),
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
                name: "model_offset".to_string(),
                choices,
            }],
            max_variants,
        },
        shape_plans: BTreeMap::new(),
        data_bindings: BTreeMap::new(),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    }
}

fn variant_scoring_controllers() -> RuntimeControllerRegistry {
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(MockController {
            id: ControllerId::new("controller:transform").unwrap(),
            handle: 1,
            emit_prediction: false,
        }))
        .unwrap();
    controllers
        .register(Box::new(VariantScoringController {
            id: ControllerId::new("controller:model").unwrap(),
            handle: 2,
        }))
        .unwrap();
    controllers
}

fn one_fold_set() -> FoldSet {
    // Single fold, train/validation disjoint (s2 trains, s1 validates). `Resampled` mode drops the
    // OOF-completeness requirement (s2 never validated) while keeping per-fold disjointness.
    FoldSet {
        id: "outer.single".to_string(),
        sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
        folds: vec![FoldAssignment {
            fold_id: FoldId::new("fold:0").unwrap(),
            train_sample_ids: vec![SampleId::new("s2").unwrap()],
            validation_sample_ids: vec![SampleId::new("s1").unwrap()],
            metadata: BTreeMap::new(),
        }],
        sample_groups: BTreeMap::new(),
        partition_mode: FoldPartitionMode::Resampled,
    }
}

fn single_fold_variant_scoring_campaign(offsets: Vec<(&str, f64)>) -> CampaignSpec {
    let mut campaign = variant_scoring_campaign(offsets);
    campaign.id = "campaign:variant.select.single.fold".to_string();
    if let Some(split) = campaign.split_invocation.as_mut() {
        split.fold_set = Some(one_fold_set());
    }
    campaign
}

// --- Multi-producer (two independent model nodes) fixtures for the >1-OOF-average refusal. ---

fn two_model_graph() -> GraphSpec {
    let data_edge = |target: &str| EdgeSpec {
        source: PortRef {
            node_id: NodeId::new("transform:snv").unwrap(),
            port_name: "x".to_string(),
        },
        target: PortRef {
            node_id: NodeId::new(target).unwrap(),
            port_name: "x".to_string(),
        },
        contract: EdgeContract {
            requires_oof: false,
            requires_fold_alignment: false,
            ..EdgeContract::new(PortKind::Data, None)
        },
    };
    GraphSpec {
        id: "g:two.model".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![
            node(
                "transform:snv",
                NodeKind::Transform,
                vec![],
                vec![port("x", PortKind::Data)],
            ),
            node(
                "model:a",
                NodeKind::Model,
                vec![port("x", PortKind::Data)],
                vec![port("pred", PortKind::Prediction)],
            ),
            node(
                "model:b",
                NodeKind::Model,
                vec![port("x", PortKind::Data)],
                vec![port("pred", PortKind::Prediction)],
            ),
        ],
        edges: vec![data_edge("model:a"), data_edge("model:b")],
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    }
}

fn two_model_manifests() -> crate::controller::ControllerRegistry {
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

fn two_model_variant_scoring_campaign(offsets: Vec<(&str, f64)>) -> CampaignSpec {
    let choices = offsets
        .into_iter()
        .map(|(label, offset)| GenerationChoice {
            label: label.to_string(),
            value: json!(label),
            param_overrides: vec![crate::generation::GenerationParamOverride {
                node_id: NodeId::new("model:a").unwrap(),
                params: BTreeMap::from([("n_components".to_string(), json!(offset))]),
            }],
        })
        .collect::<Vec<_>>();
    let max_variants = Some(choices.len());
    CampaignSpec {
        inner_cv: None,
        id: "campaign:variant.select.multi.producer".to_string(),
        root_seed: Some(7),
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
                name: "model_offset".to_string(),
                choices,
            }],
            max_variants,
        },
        shape_plans: BTreeMap::new(),
        data_bindings: BTreeMap::new(),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    }
}

fn two_model_variant_scoring_controllers() -> RuntimeControllerRegistry {
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(MockController {
            id: ControllerId::new("controller:transform").unwrap(),
            handle: 1,
            emit_prediction: false,
        }))
        .unwrap();
    controllers
        .register(Box::new(VariantScoringController {
            id: ControllerId::new("controller:model").unwrap(),
            handle: 2,
        }))
        .unwrap();
    controllers
}

#[test]
fn select_best_variant_by_cv_picks_lowest_oof_rmse_variant() {
    use crate::metrics::RegressionMetricKind;

    // Two variants over a 2-fold OOF CV: variant `accurate` predicts y_true exactly (offset 0 ->
    // RMSE 0); variant `biased` predicts y_true + 1 (offset 1 -> RMSE 1). Native SELECT must pick the
    // accurate one by its cross-fold OOF average RMSE.
    let plan = build_execution_plan(
        "plan:variant.select",
        simple_graph(),
        variant_scoring_campaign(vec![("accurate", 0.0), ("biased", 1.0)]),
        &manifests(),
    )
    .unwrap();
    assert_eq!(plan.variants.len(), 2);
    let controllers = variant_scoring_controllers();
    let run_id = RunId::new("run:variant.select").unwrap();

    let selected = select_best_variant_by_cv(
        &plan,
        &run_id,
        Some(7),
        RegressionMetricKind::Rmse,
        |variant_plan, ctx| {
            SequentialScheduler
                .execute_campaign_phase(variant_plan, &controllers, ctx, Phase::FitCv)
                .map(|_| ())
        },
    )
    .unwrap();

    let accurate_variant = plan
        .variants
        .iter()
        .find(|variant| variant.choices["model_offset"].label == "accurate")
        .unwrap();
    assert_eq!(selected, Some(accurate_variant.variant_id.clone()));
}

#[test]
fn select_best_variant_by_cv_single_variant_returns_that_variant() {
    use crate::metrics::RegressionMetricKind;

    let plan = build_execution_plan(
        "plan:variant.select.single",
        simple_graph(),
        variant_scoring_campaign(vec![("only", 1.0)]),
        &manifests(),
    )
    .unwrap();
    assert_eq!(plan.variants.len(), 1);
    let controllers = variant_scoring_controllers();
    let run_id = RunId::new("run:variant.select.single").unwrap();

    let selected = select_best_variant_by_cv(
        &plan,
        &run_id,
        Some(7),
        RegressionMetricKind::Rmse,
        |variant_plan, ctx| {
            SequentialScheduler
                .execute_campaign_phase(variant_plan, &controllers, ctx, Phase::FitCv)
                .map(|_| ())
        },
    )
    .unwrap();

    assert_eq!(selected, Some(plan.variants[0].variant_id.clone()));
}

#[test]
fn select_best_variant_by_cv_picks_highest_accuracy_variant() {
    use crate::metrics::RegressionMetricKind;

    // Accuracy maximizes (metrics.rs objective): the `accurate` variant matches the integer label
    // exactly (accuracy 1.0), `biased` is off by 1 (accuracy 0.0). Native SELECT with Accuracy must
    // pick `accurate` — proving the metric (not just RMSE) drives direction.
    let plan = build_execution_plan(
        "plan:variant.select.accuracy",
        simple_graph(),
        variant_scoring_campaign(vec![("accurate", 0.0), ("biased", 1.0)]),
        &manifests(),
    )
    .unwrap();
    assert_eq!(plan.variants.len(), 2);
    let controllers = variant_scoring_controllers();
    let run_id = RunId::new("run:variant.select.accuracy").unwrap();

    let selected = select_best_variant_by_cv(
        &plan,
        &run_id,
        Some(7),
        RegressionMetricKind::Accuracy,
        |variant_plan, ctx| {
            SequentialScheduler
                .execute_campaign_phase(variant_plan, &controllers, ctx, Phase::FitCv)
                .map(|_| ())
        },
    )
    .unwrap();

    let accurate_variant = plan
        .variants
        .iter()
        .find(|variant| variant.choices["model_offset"].label == "accurate")
        .unwrap();
    assert_eq!(selected, Some(accurate_variant.variant_id.clone()));
}

#[test]
fn select_best_variant_by_cv_no_targets_returns_none() {
    use crate::metrics::RegressionMetricKind;

    // `runtime_controllers`' model emits validation predictions but NO regression_targets, so native
    // scoring is genuinely off. The function returns Ok(None) so the caller keeps its default variant.
    let plan = build_execution_plan(
        "plan:variant.select.no.targets",
        simple_graph(),
        variant_scoring_campaign(vec![("a", 0.0), ("b", 1.0)]),
        &manifests(),
    )
    .unwrap();
    assert_eq!(plan.variants.len(), 2);
    let controllers = runtime_controllers();
    let run_id = RunId::new("run:variant.select.no.targets").unwrap();

    let selected = select_best_variant_by_cv(
        &plan,
        &run_id,
        Some(7),
        RegressionMetricKind::Rmse,
        |variant_plan, ctx| {
            SequentialScheduler
                .execute_campaign_phase(variant_plan, &controllers, ctx, Phase::FitCv)
                .map(|_| ())
        },
    )
    .unwrap();

    assert_eq!(selected, None);
}

#[test]
fn select_best_variant_by_cv_single_fold_scores_but_no_average_errors() {
    use crate::metrics::RegressionMetricKind;

    // Scoring IS on (targets emitted) but the fold set has a single fold, so `cross_fold_validation
    // _reports` skips the OOF average. Per-fold scores exist (any_scores_seen) yet no average can rank
    // the variants -> an error, distinct from the no-targets Ok(None) case.
    let plan = build_execution_plan(
        "plan:variant.select.single.fold",
        simple_graph(),
        single_fold_variant_scoring_campaign(vec![("a", 0.0), ("b", 1.0)]),
        &manifests(),
    )
    .unwrap();
    assert_eq!(plan.variants.len(), 2);
    let controllers = variant_scoring_controllers();
    let run_id = RunId::new("run:variant.select.single.fold").unwrap();

    let error = select_best_variant_by_cv(
        &plan,
        &run_id,
        Some(7),
        RegressionMetricKind::Rmse,
        |variant_plan, ctx| {
            SequentialScheduler
                .execute_campaign_phase(variant_plan, &controllers, ctx, Phase::FitCv)
                .map(|_| ())
        },
    )
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("no cross-fold OOF average"),
        "unexpected single-fold error: {error}"
    );
}

#[test]
fn select_best_variant_by_cv_rejects_multiple_prediction_producers() {
    use crate::metrics::RegressionMetricKind;

    // Two model producers each emit a cross-fold OOF average per variant, so a variant has >1 average.
    // Native SELECT needs a single score target -> it refuses to silently rank on one producer.
    let plan = build_execution_plan(
        "plan:variant.select.multi.producer",
        two_model_graph(),
        two_model_variant_scoring_campaign(vec![("a", 0.0), ("b", 1.0)]),
        &two_model_manifests(),
    )
    .unwrap();
    assert_eq!(plan.variants.len(), 2);
    let controllers = two_model_variant_scoring_controllers();
    let run_id = RunId::new("run:variant.select.multi.producer").unwrap();

    let error = select_best_variant_by_cv(
        &plan,
        &run_id,
        Some(7),
        RegressionMetricKind::Rmse,
        |variant_plan, ctx| {
            SequentialScheduler
                .execute_campaign_phase(variant_plan, &controllers, ctx, Phase::FitCv)
                .map(|_| ())
        },
    )
    .unwrap_err()
    .to_string();

    assert!(
        error.contains("multiple prediction producers"),
        "unexpected multi-producer error: {error}"
    );
}

#[test]
fn fit_view_spec_drops_excluded_samples_while_validation_keeps_them() {
    // `exclude` drops outlier samples from the TRAINING view spec (not just the
    // materialized view) but keeps them in validation/predict so OOF/test
    // coverage stays complete. three_fold_stress_set: s0..s5; fold:0
    // validation=[s0,s3], train=[s1,s2,s4,s5]. Exclude s2.
    let node_id = NodeId::new("node:model").unwrap();
    let binding = data_binding(&node_id);
    assert!(
        !binding.view_policy.include_excluded,
        "default policy must not include excluded rows"
    );
    let fold_set = three_fold_stress_set();
    let fold_id = fold_set.folds[0].fold_id.clone();
    let excluded: BTreeSet<SampleId> = [SampleId::new("s2").unwrap()].into_iter().collect();
    let empty: BTreeSet<SampleId> = BTreeSet::new();

    let fold_scope = PhaseScope {
        phase: Phase::FitCv,
        variant_id: None,
        variant: None,
        fold_id: Some(fold_id),
        seed_root: None,
    };

    // (a) Excluded sample is ABSENT from the TRAINING view spec sample_ids,
    // while the other train samples remain.
    let train_view = data_view_for_partition(
        &binding,
        Some(&fold_set),
        &fold_scope,
        DataRequestPartition::FoldTrain,
        None,
        DataViewRole::Fit,
        &excluded,
    )
    .unwrap();
    assert!(
        !train_view.include_excluded,
        "fit view must not include excluded rows"
    );
    let train_ids = train_view.sample_ids.as_ref().unwrap();
    assert!(
        !train_ids.contains(&SampleId::new("s2").unwrap()),
        "excluded s2 must be dropped from the training spec sample_ids"
    );
    assert!(
        train_ids.contains(&SampleId::new("s1").unwrap())
            && train_ids.contains(&SampleId::new("s4").unwrap())
            && train_ids.contains(&SampleId::new("s5").unwrap()),
        "non-excluded train samples must remain"
    );

    // Validation read keeps every validation sample and flags include_excluded.
    let validation_view = data_view_for_partition(
        &binding,
        Some(&fold_set),
        &fold_scope,
        DataRequestPartition::FoldValidation,
        None,
        DataViewRole::NonFit,
        &excluded,
    )
    .unwrap();
    assert!(
        validation_view.include_excluded,
        "validation read must keep excluded rows so they are still validated"
    );
    assert_eq!(
        validation_view.sample_ids,
        Some(vec![
            SampleId::new("s0").unwrap(),
            SampleId::new("s3").unwrap()
        ]),
        "validation spec is unfiltered by exclusion"
    );

    // FullTrain (refit) is a fit read: drops the excluded sample from the
    // full-train spec.
    let full_scope = PhaseScope {
        phase: Phase::Refit,
        variant_id: None,
        variant: None,
        fold_id: None,
        seed_root: None,
    };
    let full_train_view = data_view_for_partition(
        &binding,
        Some(&fold_set),
        &full_scope,
        DataRequestPartition::FullTrain,
        None,
        DataViewRole::Fit,
        &excluded,
    )
    .unwrap();
    assert!(!full_train_view.include_excluded);
    assert!(
        !full_train_view
            .sample_ids
            .as_ref()
            .unwrap()
            .contains(&SampleId::new("s2").unwrap()),
        "refit training spec drops excluded s2"
    );

    // Predict read keeps excluded; sample_ids stays None (whole dataset).
    let predict_scope = PhaseScope {
        phase: Phase::Predict,
        variant_id: None,
        variant: None,
        fold_id: None,
        seed_root: None,
    };
    let predict_view = data_view_for_partition(
        &binding,
        Some(&fold_set),
        &predict_scope,
        DataRequestPartition::Predict,
        None,
        DataViewRole::NonFit,
        &empty,
    )
    .unwrap();
    assert!(
        predict_view.include_excluded,
        "predict read must keep excluded rows so they are still predicted"
    );
}

#[test]
fn fit_influence_row_weights_match_post_exclusion_training_spec() {
    // (b) equal_sample_influence_weights row_weights length must equal the
    // post-exclusion training view, not the pre-exclusion fold train set.
    let node_id = NodeId::new("node:model").unwrap();
    let binding = data_binding(&node_id);
    let fold_set = three_fold_stress_set(); // s0..s5
    let fold_id = fold_set.folds[0].fold_id.clone();
    let train_len = fold_set.folds[0].train_sample_ids.len();
    assert!(train_len >= 2, "need a multi-sample train fold");
    let dropped = fold_set.folds[0].train_sample_ids[0].clone();
    let excluded: BTreeSet<SampleId> = [dropped.clone()].into_iter().collect();

    let scope = PhaseScope {
        phase: Phase::FitCv,
        variant_id: None,
        variant: None,
        fold_id: Some(fold_id),
        seed_root: None,
    };
    let train_view = data_view_for_partition(
        &binding,
        Some(&fold_set),
        &scope,
        DataRequestPartition::FoldTrain,
        None,
        DataViewRole::Fit,
        &excluded,
    )
    .unwrap();
    let spec_len = train_view.sample_ids.as_ref().unwrap().len();
    assert_eq!(
        spec_len,
        train_len - 1,
        "training spec must drop exactly the one excluded sample"
    );
    assert!(!train_view.sample_ids.as_ref().unwrap().contains(&dropped));

    let mut data_views = BTreeMap::new();
    data_views.insert("x".to_string(), train_view);
    let weights = equal_sample_influence_weights(&data_views).expect("weights derived");
    assert_eq!(
        weights.len(),
        spec_len,
        "row_weights length must equal the post-exclusion training spec"
    );
}

#[test]
fn exclusion_is_sample_local_across_relation_rows() {
    // (c) A sample with one excluded relation row and one non-excluded row is
    // fully dropped from training (sample-local exclusion).
    let mut base = SampleRelation::new(
        ObservationId::new("obs.s2.base").unwrap(),
        SampleId::new("s2").unwrap(),
    );
    base.excluded = false;
    let mut rep = SampleRelation::new(
        ObservationId::new("obs.s2.rep1").unwrap(),
        SampleId::new("s2").unwrap(),
    );
    rep.excluded = true; // only the second row is excluded
    let kept = SampleRelation::new(
        ObservationId::new("obs.s1.base").unwrap(),
        SampleId::new("s1").unwrap(),
    );
    let relations = SampleRelationSet {
        records: vec![base, rep, kept],
    };
    let excluded = relations.excluded_sample_ids();
    assert!(
        excluded.contains(&SampleId::new("s2").unwrap()),
        "a sample with ANY excluded row is excluded sample-locally"
    );
    assert!(!excluded.contains(&SampleId::new("s1").unwrap()));

    let node_id = NodeId::new("node:model").unwrap();
    let binding = data_binding(&node_id);
    let fold_set = three_fold_stress_set();
    let fold_id = fold_set.folds[0].fold_id.clone(); // train=[s1,s2,s4,s5]
    let scope = PhaseScope {
        phase: Phase::FitCv,
        variant_id: None,
        variant: None,
        fold_id: Some(fold_id),
        seed_root: None,
    };
    let train_view = data_view_for_partition(
        &binding,
        Some(&fold_set),
        &scope,
        DataRequestPartition::FoldTrain,
        None,
        DataViewRole::Fit,
        &excluded,
    )
    .unwrap();
    assert!(
        !train_view
            .sample_ids
            .as_ref()
            .unwrap()
            .contains(&SampleId::new("s2").unwrap()),
        "s2 must be fully dropped from training even though one of its rows is not excluded"
    );
}

#[test]
fn by_metadata_and_by_tag_branch_selectors_reach_the_provider_view_spec() {
    // The metadata/tag branch selector must survive the scheduler's
    // `data_view_for_partition` path and arrive intact on the
    // `DataProviderViewSpec.branch_view` that is handed to the provider's
    // `make_view` (where dag-ml-data's `filter_relations` matches it natively).
    use crate::data::{BranchViewMode, BranchViewPlan, DataViewSelector};

    let node_id = NodeId::new("node:model").unwrap();
    let binding = data_binding(&node_id);
    let empty: BTreeSet<SampleId> = BTreeSet::new();
    let scope = PhaseScope {
        phase: Phase::Predict,
        variant_id: None,
        variant: None,
        fold_id: None,
        seed_root: None,
    };

    let metadata_branch = BranchViewPlan {
        view_id: "branch_view:group_a".to_string(),
        branch_id: "branch:group_a".to_string(),
        mode: BranchViewMode::ByMetadata,
        selector: DataViewSelector {
            metadata: BTreeMap::from([("group".to_string(), serde_json::json!("A"))]),
            ..Default::default()
        },
        allow_overlap: false,
        metadata: BTreeMap::new(),
    };
    let metadata_view = data_view_for_partition(
        &binding,
        None,
        &scope,
        DataRequestPartition::Predict,
        Some(&metadata_branch),
        DataViewRole::NonFit,
        &empty,
    )
    .unwrap();
    let carried = metadata_view
        .branch_view
        .as_ref()
        .expect("by_metadata selector must reach the provider view spec");
    assert_eq!(carried.mode, BranchViewMode::ByMetadata);
    assert_eq!(
        carried.selector.metadata.get("group"),
        Some(&serde_json::json!("A")),
        "by_metadata selector value must reach the provider unchanged"
    );

    let tag_branch = BranchViewPlan {
        view_id: "branch_view:clean".to_string(),
        branch_id: "branch:clean".to_string(),
        mode: BranchViewMode::ByTag,
        selector: DataViewSelector {
            tags: vec!["clean".to_string()],
            ..Default::default()
        },
        allow_overlap: false,
        metadata: BTreeMap::new(),
    };
    let tag_view = data_view_for_partition(
        &binding,
        None,
        &scope,
        DataRequestPartition::Predict,
        Some(&tag_branch),
        DataViewRole::NonFit,
        &empty,
    )
    .unwrap();
    let carried_tags = tag_view
        .branch_view
        .as_ref()
        .expect("by_tag selector must reach the provider view spec");
    assert_eq!(carried_tags.mode, BranchViewMode::ByTag);
    assert_eq!(
        carried_tags.selector.tags,
        vec!["clean".to_string()],
        "by_tag selector must reach the provider unchanged"
    );
    // The spec itself must validate (the branch view validation runs here too).
    metadata_view.validate().unwrap();
    tag_view.validate().unwrap();
}

// ----- Slice 2: data-aware fan-out per-branch FIT_CV scoping -----

/// One recorded FIT_CV task observation:
/// `(node_id, fold_id, branch-view "site" value, validation sample ids)`.
type BranchScopeObservation = (String, String, Option<serde_json::Value>, Vec<String>);

/// Records, per (node_id, fold_id), the branch-view selector value and the
/// validation-view sample ids the runtime hands to the controller, then emits a
/// validation OOF block scoped to ITS PARTITION. A real branch model node only
/// sees the samples the data provider returns after applying the branch_view
/// metadata filter (the partition ∩ fold-validation intersection), so this mock
/// reproduces that by intersecting the fold-validation ids with the samples whose
/// recorded site equals the node's branch_view selector value — and detects the
/// empty intersection explicitly rather than silently emitting nothing.
struct BranchScopeRecordingController {
    id: ControllerId,
    handle: u64,
    /// sample id -> site value, the membership the data provider would filter by.
    sample_sites: BTreeMap<String, String>,
    /// When true, an empty partition ∩ fold is a hard error (mirrors the
    /// dag-ml-data provider's "data view selected no coordinator relations").
    /// When false, the branch+fold is skipped with no OOF block (the alternative
    /// explicit handling: never silently duplicate/mis-cover).
    error_on_empty_intersection: bool,
    seen: Arc<Mutex<Vec<BranchScopeObservation>>>,
}

impl RuntimeController for BranchScopeRecordingController {
    fn controller_id(&self) -> &ControllerId {
        &self.id
    }

    fn invoke(&self, task: &NodeTask) -> Result<NodeResult> {
        let node_id = task.node_plan.node_id.to_string();
        let fold_label = task
            .fold_id
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_else(|| "nofold".to_string());
        // The validation companion view carries this node's branch_view selector.
        let validation_view = task
            .data_views
            .iter()
            .find(|(_, view)| view.partition == DataRequestPartition::FoldValidation)
            .map(|(_, view)| view);
        let branch_value = validation_view
            .and_then(|view| view.branch_view.as_ref())
            .and_then(|branch| branch.selector.metadata.get("site").cloned());
        let branch_site = branch_value.as_ref().and_then(|value| value.as_str());
        // The fold's full validation sample ids carried by the spec.
        let fold_validation_ids: Vec<String> = validation_view
            .and_then(|view| view.sample_ids.clone())
            .unwrap_or_default()
            .iter()
            .map(ToString::to_string)
            .collect();
        // Partition ∩ fold-validation: only the fold-validation samples whose site
        // matches this branch's selector (what the data provider would return).
        let partition_ids: Vec<String> = match branch_site {
            Some(site) => fold_validation_ids
                .iter()
                .filter(|sample| {
                    self.sample_sites
                        .get(*sample)
                        .map(|s| s == site)
                        .unwrap_or(false)
                })
                .cloned()
                .collect(),
            None => fold_validation_ids.clone(),
        };
        self.seen.lock().unwrap().push((
            node_id,
            fold_label.clone(),
            branch_value.clone(),
            partition_ids.clone(),
        ));

        let prediction_output = HandleRef {
            handle: self.handle,
            kind: HandleKind::Prediction,
            owner_controller: self.id.clone(),
        };
        let data_output = HandleRef {
            handle: self.handle,
            kind: HandleKind::Data,
            owner_controller: self.id.clone(),
        };
        // Explicit empty partition ∩ fold handling: error or skip, never emit a
        // silently-empty OOF block that would mis-cover the partition. When not
        // erroring, the empty (branch, fold) is simply skipped below (no block).
        if task.phase == Phase::FitCv
            && task.fold_id.is_some()
            && partition_ids.is_empty()
            && self.error_on_empty_intersection
        {
            return Err(DagMlError::RuntimeValidation(format!(
                "branch node `{}` has no samples in fold {} for its partition (empty \
                 partition ∩ fold)",
                task.node_plan.node_id, fold_label
            )));
        }
        let predictions = if task.phase == Phase::FitCv && !partition_ids.is_empty() {
            vec![PredictionBlock {
                prediction_id: Some(format!("pred:{}", task.node_plan.node_id)),
                producer_node: task.node_plan.node_id.clone(),
                partition: PredictionPartition::Validation,
                fold_id: task.fold_id.clone(),
                sample_ids: partition_ids
                    .iter()
                    .map(|s| SampleId::new(s).unwrap())
                    .collect(),
                values: vec![vec![1.0]; partition_ids.len()],
                target_names: vec!["y".to_string()],
            }]
        } else {
            Vec::new()
        };
        Ok(NodeResult {
            node_id: task.node_plan.node_id.clone(),
            outputs: BTreeMap::from([
                ("pred".to_string(), prediction_output.clone()),
                ("oof".to_string(), prediction_output),
                ("x".to_string(), data_output),
            ]),
            predictions,
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
            explanations: Vec::new(),
            shape_deltas: Vec::new(),
            artifacts: Vec::new(),
            artifact_handles: BTreeMap::new(),
            fit_influence_diagnostics: Vec::new(),
            regression_targets: Vec::new(),
            lineage: LineageRecord {
                record_id: LineageId::new(format!(
                    "lineage:{}:{fold_label}",
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

/// A model node carrying a `dsl_branch_view_plan` by_metadata selector for one
/// site value — exactly the shape `fan_out_data_aware_branches` + compile emit.
fn branch_model_node(node_id: &str, site: &str) -> NodeSpec {
    use crate::data::{BranchViewMode, BranchViewPlan, DataViewSelector};
    let plan = BranchViewPlan {
        view_id: format!("branch_view:per_site__{site}"),
        branch_id: format!("per_site__{site}"),
        mode: BranchViewMode::ByMetadata,
        selector: DataViewSelector {
            metadata: BTreeMap::from([("site".to_string(), serde_json::json!(site))]),
            ..Default::default()
        },
        allow_overlap: false,
        metadata: BTreeMap::new(),
    };
    let mut node = node(
        node_id,
        NodeKind::Model,
        vec![port("x", PortKind::Data)],
        vec![port("oof", PortKind::Prediction)],
    );
    node.metadata.insert(
        "dsl_branch_view_plan".to_string(),
        serde_json::to_value(&plan).unwrap(),
    );
    node
}

#[test]
fn fanned_out_branches_each_fit_cv_only_their_partition() {
    // Two fanned-out branch model nodes (one per discovered site A/B), each
    // scoped to its partition by a dsl_branch_view_plan in node metadata — the
    // shape the data-aware fan-out + compile produce. A 2-fold KFold over four
    // samples drives FIT_CV; we assert each branch node, across both folds,
    // receives a validation view carrying ITS OWN branch_view selector and the
    // fold's validation samples (the intersection that yields per-partition OOF).
    let node_a = branch_model_node("model:site__A", "A");
    let node_b = branch_model_node("model:site__B", "B");
    let graph = GraphSpec {
        id: "graph:fanout.scope".to_string(),
        interface: GraphInterface::default(),
        nodes: vec![node_a, node_b],
        edges: Vec::new(),
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    };

    let id_a = NodeId::new("model:site__A").unwrap();
    let id_b = NodeId::new("model:site__B").unwrap();
    let samples: Vec<SampleId> = ["sample:1", "sample:2", "sample:3", "sample:4"]
        .iter()
        .map(|s| SampleId::new(*s).unwrap())
        .collect();
    let fold_set = FoldSet {
        id: "folds:fanout.scope".to_string(),
        sample_ids: samples.clone(),
        folds: vec![
            FoldAssignment {
                fold_id: FoldId::new("fold:0").unwrap(),
                train_sample_ids: vec![samples[2].clone(), samples[3].clone()],
                validation_sample_ids: vec![samples[0].clone(), samples[1].clone()],
                metadata: BTreeMap::new(),
            },
            FoldAssignment {
                fold_id: FoldId::new("fold:1").unwrap(),
                train_sample_ids: vec![samples[0].clone(), samples[1].clone()],
                validation_sample_ids: vec![samples[2].clone(), samples[3].clone()],
                metadata: BTreeMap::new(),
            },
        ],
        sample_groups: BTreeMap::new(),
        partition_mode: FoldPartitionMode::Partition,
    };
    let campaign = CampaignSpec {
        inner_cv: None,
        id: "campaign:fanout.scope".to_string(),
        root_seed: Some(7),
        leakage_policy: Default::default(),
        aggregation_policy: Default::default(),
        split_invocation: Some(SplitInvocation {
            id: "split:fanout.scope".to_string(),
            controller_id: None,
            leakage_policy: Default::default(),
            params: BTreeMap::new(),
            fold_set: Some(fold_set),
        }),
        generation: Default::default(),
        shape_plans: BTreeMap::new(),
        data_bindings: BTreeMap::from([
            (id_a.clone(), vec![data_binding(&id_a)]),
            (id_b.clone(), vec![data_binding(&id_b)]),
        ]),
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    };
    let mut registry = ControllerRegistry::new();
    registry
        .register(controller_manifest("controller:model", NodeKind::Model))
        .unwrap();
    let plan = build_execution_plan("plan:fanout.scope", graph, campaign, &registry).unwrap();

    // The envelope carries 4 samples across two sites; require_relations is set,
    // so registering it is enough for the InMemoryDataProvider to materialize.
    let envelope = sample_relations_envelope(&[
        ("sample:1", "A"),
        ("sample:2", "B"),
        ("sample:3", "A"),
        ("sample:4", "B"),
    ]);
    let data_provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data").unwrap(),
        envelope,
    )
    .unwrap();

    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(BranchScopeRecordingController {
            id: ControllerId::new("controller:model").unwrap(),
            handle: 1,
            sample_sites: BTreeMap::from([
                ("sample:1".to_string(), "A".to_string()),
                ("sample:2".to_string(), "B".to_string()),
                ("sample:3".to_string(), "A".to_string()),
                ("sample:4".to_string(), "B".to_string()),
            ]),
            error_on_empty_intersection: false,
            seen: Arc::clone(&seen),
        }))
        .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:fanout.scope").unwrap(), Some(7));

    SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &data_provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap();

    let seen = seen.lock().unwrap().clone();
    // Two branch nodes x two folds = four FIT_CV tasks.
    assert_eq!(seen.len(), 4, "expected one task per (branch, fold)");
    // Every branch-A task is scoped to site A; every branch-B task to site B.
    for (node_id, _fold, branch_value, _val_ids) in &seen {
        let expected = if node_id.ends_with("__A") { "A" } else { "B" };
        assert_eq!(
            branch_value.as_ref(),
            Some(&serde_json::json!(expected)),
            "node `{node_id}` must be scoped to its own partition `{expected}`"
        );
    }
    // Realistic per-partition OOF: each branch validates ONLY its partition's
    // samples (partition ∩ fold-validation), never the full universe. Site A owns
    // {sample:1, sample:3}; site B owns {sample:2, sample:4}.
    let oof_blocks = ctx.prediction_store.blocks();
    assert_eq!(oof_blocks.len(), 4, "one OOF block per (branch, fold)");
    for block in oof_blocks {
        assert_eq!(block.partition, PredictionPartition::Validation);
        assert!(
            !block.sample_ids.is_empty(),
            "a non-empty partition ∩ fold must produce a non-empty OOF block"
        );
    }
    let expected_partition = BTreeMap::from([
        (
            "model:site__A",
            vec!["sample:1".to_string(), "sample:3".to_string()],
        ),
        (
            "model:site__B",
            vec!["sample:2".to_string(), "sample:4".to_string()],
        ),
    ]);
    for (node, expected) in &expected_partition {
        let mut ids: Vec<String> = oof_blocks
            .iter()
            .filter(|block| block.producer_node.as_str() == *node)
            .flat_map(|block| block.sample_ids.iter().map(ToString::to_string))
            .collect();
        ids.sort();
        assert_eq!(
            &ids, expected,
            "branch `{node}` must validate ONLY its own partition's samples across folds"
        );
    }
}

/// Build the 3-site plan + provider used by the empty-intersection tests. Site C
/// has a single sample (sample:4) so one fold's validation set contains no C
/// samples → an empty partition ∩ fold for branch C in that fold.
fn empty_intersection_plan_and_provider() -> (ExecutionPlan, InMemoryDataProvider) {
    let nodes = vec![
        branch_model_node("model:site__A", "A"),
        branch_model_node("model:site__B", "B"),
        branch_model_node("model:site__C", "C"),
    ];
    let graph = GraphSpec {
        id: "graph:fanout.empty".to_string(),
        interface: GraphInterface::default(),
        nodes,
        edges: Vec::new(),
        search_space_fingerprint: None,
        metadata: BTreeMap::new(),
    };
    let samples: Vec<SampleId> = ["sample:1", "sample:2", "sample:3", "sample:4"]
        .iter()
        .map(|s| SampleId::new(*s).unwrap())
        .collect();
    let fold_set = FoldSet {
        id: "folds:fanout.empty".to_string(),
        sample_ids: samples.clone(),
        folds: vec![
            FoldAssignment {
                fold_id: FoldId::new("fold:0").unwrap(),
                train_sample_ids: vec![samples[2].clone(), samples[3].clone()],
                // val = {sample:1 (A), sample:2 (B)} -> NO C samples here.
                validation_sample_ids: vec![samples[0].clone(), samples[1].clone()],
                metadata: BTreeMap::new(),
            },
            FoldAssignment {
                fold_id: FoldId::new("fold:1").unwrap(),
                train_sample_ids: vec![samples[0].clone(), samples[1].clone()],
                // val = {sample:3 (A), sample:4 (C)}.
                validation_sample_ids: vec![samples[2].clone(), samples[3].clone()],
                metadata: BTreeMap::new(),
            },
        ],
        sample_groups: BTreeMap::new(),
        partition_mode: FoldPartitionMode::Partition,
    };
    let bindings = ["model:site__A", "model:site__B", "model:site__C"]
        .iter()
        .map(|id| {
            let node_id = NodeId::new(*id).unwrap();
            (node_id.clone(), vec![data_binding(&node_id)])
        })
        .collect::<BTreeMap<_, _>>();
    let campaign = CampaignSpec {
        inner_cv: None,
        id: "campaign:fanout.empty".to_string(),
        root_seed: Some(7),
        leakage_policy: Default::default(),
        aggregation_policy: Default::default(),
        split_invocation: Some(SplitInvocation {
            id: "split:fanout.empty".to_string(),
            controller_id: None,
            leakage_policy: Default::default(),
            params: BTreeMap::new(),
            fold_set: Some(fold_set),
        }),
        generation: Default::default(),
        shape_plans: BTreeMap::new(),
        data_bindings: bindings,
        branch_view_plans: Vec::new(),
        metadata: BTreeMap::new(),
    };
    let mut registry = ControllerRegistry::new();
    registry
        .register(controller_manifest("controller:model", NodeKind::Model))
        .unwrap();
    let plan = build_execution_plan("plan:fanout.empty", graph, campaign, &registry).unwrap();
    let envelope = sample_relations_envelope(&[
        ("sample:1", "A"),
        ("sample:2", "B"),
        ("sample:3", "A"),
        ("sample:4", "C"),
    ]);
    let provider = InMemoryDataProvider::with_envelope(
        ControllerId::new("controller:data").unwrap(),
        envelope,
    )
    .unwrap();
    (plan, provider)
}

fn empty_intersection_sample_sites() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("sample:1".to_string(), "A".to_string()),
        ("sample:2".to_string(), "B".to_string()),
        ("sample:3".to_string(), "A".to_string()),
        ("sample:4".to_string(), "C".to_string()),
    ])
}

#[test]
fn empty_partition_intersection_is_skipped_with_no_silent_miscoverage() {
    // Skip mode: branch C has NO samples in fold:0 (its partition ∩ fold is
    // empty). The (C, fold:0) OOF is skipped — never a silently-empty block — and
    // C still validates its sample in fold:1, so coverage is correct, not dropped.
    let (plan, provider) = empty_intersection_plan_and_provider();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(BranchScopeRecordingController {
            id: ControllerId::new("controller:model").unwrap(),
            handle: 1,
            sample_sites: empty_intersection_sample_sites(),
            error_on_empty_intersection: false,
            seen: Arc::clone(&seen),
        }))
        .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:fanout.empty.skip").unwrap(), Some(7));

    SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap();

    let oof_blocks = ctx.prediction_store.blocks();
    // Branch C produced exactly one OOF block (fold:1, sample:4); fold:0 skipped.
    let c_ids: Vec<String> = oof_blocks
        .iter()
        .filter(|block| block.producer_node.as_str() == "model:site__C")
        .flat_map(|block| block.sample_ids.iter().map(ToString::to_string))
        .collect();
    assert_eq!(
        c_ids,
        vec!["sample:4".to_string()],
        "branch C must cover only its present sample, with the empty fold skipped"
    );
    // No empty OOF blocks were emitted anywhere.
    assert!(
        oof_blocks.iter().all(|block| !block.sample_ids.is_empty()),
        "no silently-empty OOF block may be emitted for an empty partition ∩ fold"
    );
}

#[test]
fn empty_partition_intersection_can_raise_a_clear_error() {
    // Error mode: the empty partition ∩ fold for branch C in fold:0 raises a
    // clear, explicit error rather than silently mis-covering.
    let (plan, provider) = empty_intersection_plan_and_provider();
    let mut controllers = RuntimeControllerRegistry::new();
    controllers
        .register(Box::new(BranchScopeRecordingController {
            id: ControllerId::new("controller:model").unwrap(),
            handle: 1,
            sample_sites: empty_intersection_sample_sites(),
            error_on_empty_intersection: true,
            seen: Arc::new(Mutex::new(Vec::new())),
        }))
        .unwrap();
    let mut ctx = RunContext::new(RunId::new("run:fanout.empty.error").unwrap(), Some(7));

    let error = SequentialScheduler
        .execute_campaign_phase_with_data_provider(
            &plan,
            &controllers,
            &provider,
            &mut ctx,
            Phase::FitCv,
        )
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("empty partition ∩ fold"),
        "empty intersection must surface a clear error: {error}"
    );
}

fn sample_relations_envelope(rows: &[(&str, &str)]) -> ExternalDataPlanEnvelope {
    let records = rows
        .iter()
        .map(|(sample, site)| {
            let mut relation = SampleRelation::new(
                ObservationId::new(format!("obs:{}", sample.replace(':', "."))).unwrap(),
                SampleId::new(*sample).unwrap(),
            );
            relation
                .metadata
                .insert("site".to_string(), serde_json::json!(site));
            relation
        })
        .collect::<Vec<_>>();
    let relations = SampleRelationSet { records };
    relations.validate().unwrap();
    // Match the data_binding() helper's fingerprints so the provider accepts the
    // binding; require_relations is satisfied by coordinator_relations presence.
    ExternalDataPlanEnvelope {
        schema_version: crate::data::EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION,
        schema_fingerprint: "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b"
            .to_string(),
        plan_fingerprint: "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d"
            .to_string(),
        relation_fingerprint: Some(
            "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10".to_string(),
        ),
        coordinator_relations: Some(relations),
    }
}
