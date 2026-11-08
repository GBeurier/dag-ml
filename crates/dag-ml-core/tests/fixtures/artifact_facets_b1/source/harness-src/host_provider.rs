// DERIVED COPY (host adapter, no Core crate dependency): lines 1-985 of
// nirs4all-core/bindings/rust/nirs4all/src/io_training.rs, git blob 830f5b9eab633e4ba2075d4fffeb83605dbfe54d
// (file sha256 5d8221768a5a9ea06fcd08814833a67763b1b76f4cbb827fc80cd836c819d50d, last commit d5533fd6bd355d58b866359c5c796bd70b870d2a).
// Only mechanical edits: removed two crate-local archive/runtime imports (lines 35-36), crate alias
// names dag_ml_data_crate/dag_ml_data_provider_crate -> dag_ml_data/dag_ml_data_provider, pub(crate) -> pub.
// Everything else is verbatim; no numerics, only identity-keyed IO-package -> DAG-ML provider glue.
#![allow(unused_imports, dead_code)]
//! Typed IO `DatasetPackage` to DAG-ML/Methods training composition.
//!
//! IO remains the authority for package materialization and numeric buffers;
//! DAG-ML remains the authority for folds, training, selection and replay.
//! This module only adapts their public provider contracts inside the portable
//! aggregate.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Mutex;

use dag_ml_core::{
    build_archive_v2_native_portable_payloads,
    calibrate_attached_training_replay_with_derived_context, data_binding_requirement_key,
    execute_attached_training_replay, execute_training, ArtifactLoadMode,
    AttachedTrainingReplayInput, BundleDataRequirement, BundleId, ConformalCalibration,
    ConformalCalibrationTruth, ConformalMultiTargetPolicy, ConformalSmallSamplePolicy,
    ControllerId, DataBinding, DataMaterializationRequest, DataProviderViewSpec, DataViewRequest,
    EntityUnitLevel, ExternalDataPlanEnvelope, FittedArtifactMode, GraphSpec, HandleKind,
    HandleRef, InMemoryArtifactStore, MethodsPlsController, MethodsPlsData, MethodsPlsDataRequest,
    MethodsPlsDataset, MethodsPlsMatrix, Phase, RunId, RuntimeControllerRegistry,
    RuntimeDataProvider, SampleRelation, SampleRelationSet, TrainingDataIdentity,
    TrainingExecutionInput, TrainingInfluenceManifest, TrainingOutcome, TrainingReplayOutcome,
    TrainingReplayRequest, TrainingRequest, TRAINING_REPLAY_REQUEST_SCHEMA_VERSION,
    TRAINING_REQUEST_SCHEMA_VERSION,
};
use dag_ml_data::{
    CoordinatorDataMaterializationRequest, CoordinatorHandleKind, DataView, SampleId as IoSampleId,
    SourceId as IoSourceId,
};
use dag_ml_data_provider::DagMlDataProvider;
pub use nirs4all_io_dagml::DatasetPackage;
use nirs4all_io_dagml::PackageProvider;


#[derive(Clone, Debug, PartialEq)]
struct ViewSlot {
    node_id: dag_ml_core::NodeId,
    input_name: String,
    phase: Phase,
    variant_id: Option<dag_ml_core::VariantId>,
    fold_id: Option<dag_ml_core::FoldId>,
    view: DataProviderViewSpec,
    handle: u64,
}

#[derive(Default)]
struct ProviderHandles {
    data: BTreeSet<u64>,
    views: Vec<ViewSlot>,
}

/// Runtime adapter over one exact numeric source selected from an IO package.
///
/// The adapter does not own or reinterpret feature buffers. It forwards every
/// scheduler-created identity view to IO's `PackageProvider`, projects the
/// resulting typed `f64` block, and releases all provider handles explicitly.
pub struct DatasetPackageMethodsProvider {
    provider: PackageProvider,
    external_envelope: ExternalDataPlanEnvelope,
    relations: SampleRelationSet,
    source_id: String,
    target_masks: Option<BTreeMap<String, Vec<bool>>>,
    handles: Mutex<ProviderHandles>,
}

impl DatasetPackageMethodsProvider {
    /// Select one source from a dense numeric package. Fusion, N-D tensors and
    /// named processing stacks remain fail-closed in the IO owner.
    pub fn new(package: &DatasetPackage, source_id: &str) -> Result<Self, String> {
        let result = Self::for_product_pipeline(package, source_id, false)?;
        if result
            .target_masks
            .as_ref()
            .is_some_and(|rows| rows.values().any(|row| row.contains(&false)))
        {
            return Err("Legacy complete-target workflow refuses partial target masks".into());
        }
        Ok(result)
    }

    pub fn for_product_pipeline(
        package: &DatasetPackage,
        source_id: &str,
        classification: bool,
    ) -> Result<Self, String> {
        if package.task_type
            != if classification {
                "classification"
            } else {
                "regression"
            }
        {
            return Err(format!(
                "Core native pipeline role/task_type disagree, got `{}`",
                package.task_type
            ));
        }
        let provider = PackageProvider::from_package_source(package, source_id)
            .map_err(|error| error.to_string())?;
        if provider.target_names().is_empty() || provider.target_ids().is_empty() {
            return Err("Core Methods package training requires numeric targets".to_string());
        }
        let relations = convert_relations(
            provider
                .envelope()
                .coordinator_relations
                .as_ref()
                .ok_or("Core Methods package training requires coordinator relations")?,
        )?;
        let mut target_masks = None;
        for audit in &package.audits {
            if let Some(mask) = audit.get("target_mask").filter(|mask| !mask.is_null()) {
                let ids: Vec<String> = serde_json::from_value(audit["input_sample_ids"].clone())
                    .map_err(|e| e.to_string())?;
                let rows = mask["values"]
                    .as_array()
                    .ok_or("Invalid native target mask rows")?;
                if rows.len() != ids.len() {
                    return Err("Native mask/sample dimensions disagree".into());
                }
                let mut keyed = BTreeMap::new();
                for (id, row) in ids.into_iter().zip(rows) {
                    let cells = if let Some(value) = row.as_bool() {
                        vec![value]
                    } else {
                        row.as_array()
                            .ok_or("Invalid native target mask row")?
                            .iter()
                            .map(|value| value.as_bool().ok_or("Invalid native target mask cell"))
                            .collect::<Result<Vec<_>, _>>()?
                    };
                    if cells.len() != provider.target_names().len()
                        || keyed.insert(id, cells).is_some()
                    {
                        return Err("Native target mask schema or identity mismatch".into());
                    }
                }
                if keyed.len() != relations.records.len()
                    || relations
                        .records
                        .iter()
                        .any(|r| !keyed.contains_key(r.sample_id.as_str()))
                {
                    return Err("Native target mask identities disagree with provider".into());
                }
                target_masks = Some(keyed);
            }
        }
        let relation_fingerprint = relations.fingerprint().map_err(|error| error.to_string())?;
        let mut external_envelope = ExternalDataPlanEnvelope {
            schema_version: 1,
            schema_fingerprint: provider.envelope().schema_fingerprint.clone(),
            plan_fingerprint: provider.envelope().plan_fingerprint.clone(),
            relation_fingerprint: Some(relation_fingerprint),
            data_content_fingerprint: provider.envelope().data_content_fingerprint.clone(),
            target_content_fingerprint: provider.envelope().target_content_fingerprint.clone(),
            coordinator_relations: Some(relations.clone()),
            predict_cohort: None,
        };
        if let Some(masks) = &target_masks {
            use sha2::{Digest, Sha256};
            let identity = serde_json::to_vec(&serde_json::json!({"target_content_fingerprint":external_envelope.target_content_fingerprint,"target_names":provider.target_names(),"observed_by_sample":masks})).map_err(|e|e.to_string())?;
            external_envelope.target_content_fingerprint =
                Some(format!("{:x}", Sha256::digest(identity)));
        }
        external_envelope
            .validate()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            provider,
            external_envelope,
            relations,
            source_id: source_id.to_string(),
            target_masks,
            handles: Mutex::new(ProviderHandles::default()),
        })
    }

    pub fn external_envelope(&self) -> &ExternalDataPlanEnvelope {
        &self.external_envelope
    }

    pub fn relations(&self) -> &SampleRelationSet {
        &self.relations
    }

    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    pub fn bind_replay_requirement(
        &mut self,
        requirement: &BundleDataRequirement,
    ) -> Result<(), String> {
        requirement.validate().map_err(|error| error.to_string())?;
        if requirement.feature_set_id.as_deref() != Some(self.provider.feature_set_id())
            || requirement.output_representation
                != self.provider.envelope().plan.output_representation.as_str()
        {
            return Err(format!(
                "Core calibration source `{}` is incompatible with signed data requirement `{}`",
                self.source_id,
                requirement.key()
            ));
        }
        // The bundle owns the training-time structural contract. A held-out
        // package naturally has different concrete relation/fold metadata,
        // while IO remains authoritative for its buffers and content proofs.
        self.external_envelope.schema_fingerprint = requirement.schema_fingerprint.clone();
        self.external_envelope.plan_fingerprint = requirement.plan_fingerprint.clone();
        self.external_envelope
            .validate()
            .map_err(|error| error.to_string())
    }

    /// Explicitly release all IO provider handles. `Drop` invokes the same
    /// operation, so early-return and error paths are deterministic too.
    pub fn release_all(&self) {
        let Ok(mut handles) = self.handles.lock() else {
            return;
        };
        for slot in handles.views.drain(..).rev() {
            self.provider.release(slot.handle);
        }
        for handle in std::mem::take(&mut handles.data).into_iter().rev() {
            self.provider.release(handle);
        }
    }

    fn validate_binding(&self, binding: &DataBinding) -> Result<(), String> {
        binding
            .validate_envelope(&self.external_envelope)
            .map_err(|error| error.to_string())?;
        if binding.feature_set_id() != self.provider.feature_set_id()
            || binding.source_ids.as_slice() != [self.source_id.as_str()]
            || binding.output_representation
                != self.provider.envelope().plan.output_representation.as_str()
        {
            return Err(format!(
                "Core Methods package binding `{}` must select only IO source `{}` and its exact output representation",
                data_binding_requirement_key(&binding.node_id, &binding.input_name),
                self.source_id
            ));
        }
        Ok(())
    }

    fn view_handle(
        &self,
        request: &MethodsPlsDataRequest,
        view: &DataProviderViewSpec,
    ) -> Result<u64, String> {
        let handles = self
            .handles
            .lock()
            .map_err(|_| "Core IO provider handle registry is poisoned".to_string())?;
        handles
            .views
            .iter()
            .rev()
            .find(|slot| {
                slot.node_id == request.node_id
                    && slot.input_name == request.binding.input_name
                    && slot.phase == request.phase
                    && slot.variant_id == request.variant_id
                    && slot.fold_id == request.fold_id
                    && slot.view == *view
            })
            .map(|slot| slot.handle)
            .ok_or_else(|| {
                format!(
                    "Core IO provider has no scheduler-created view for `{}.{}` {:?}",
                    request.node_id, request.binding.input_name, request.phase
                )
            })
    }

    fn project_dataset(
        &self,
        request: &MethodsPlsDataRequest,
        view: &DataProviderViewSpec,
        require_targets: bool,
    ) -> Result<MethodsPlsDataset, String> {
        let view_handle = self.view_handle(request, view)?;
        let feature = self
            .provider
            .feature_block_f64(view_handle)
            .map_err(|error| error.to_string())?;
        if feature.feature_set_id != self.provider.feature_set_id()
            || feature.observation_ids.len() != feature.sample_ids.len()
            || feature
                .sample_ids
                .len()
                .checked_mul(feature.feature_names.len())
                != Some(feature.values.len())
        {
            return Err("Core IO provider returned an invalid typed feature block".to_string());
        }
        let sample_ids = feature
            .sample_ids
            .iter()
            .map(|id| dag_ml_core::SampleId::new(id.as_str()).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let target_names = self.provider.target_names().to_vec();
        let y = if require_targets {
            Some(self.project_targets(view_handle, &feature.sample_ids, &target_names)?)
        } else {
            None
        };
        let y_validity_masks = if require_targets {
            self.target_masks
                .as_ref()
                .map(|masks| {
                    sample_ids
                        .iter()
                        .map(|id| {
                            masks.get(id.as_str()).cloned().ok_or_else(|| {
                                "Native target mask missing sample identity".to_string()
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
        } else {
            None
        };
        let dataset = MethodsPlsDataset {
            y_validity_masks,
            sample_ids,
            x: MethodsPlsMatrix {
                rows: feature.sample_ids.len(),
                cols: feature.feature_names.len(),
                values: feature.values,
            },
            y,
            target_names,
        };
        dataset
            .validate("IO DatasetPackage", require_targets)
            .map_err(|error| error.to_string())?;
        Ok(dataset)
    }

    fn project_targets(
        &self,
        view_handle: u64,
        sample_ids: &[IoSampleId],
        target_names: &[String],
    ) -> Result<MethodsPlsMatrix, String> {
        let target_ids = self.provider.target_ids();
        let values = if target_ids.len() == 1 && target_names.len() > 1 {
            let block = self
                .provider
                .target_block(view_handle, &target_ids[0])
                .map_err(|error| error.to_string())?;
            validate_target_samples(sample_ids, &block.sample_ids)?;
            let mut values = Vec::with_capacity(sample_ids.len() * target_names.len());
            for value in &block.values {
                let row = value
                    .as_array()
                    .ok_or("Core IO provider multi-target block contains a non-array row")?;
                if row.len() != target_names.len() {
                    return Err(
                        "Core IO provider multi-target block width differs from target names"
                            .to_string(),
                    );
                }
                for cell in row {
                    values.push(finite_number(cell, "multi-target cell")?);
                }
            }
            values
        } else {
            if target_ids.len() != target_names.len() {
                return Err(
                    "Core IO provider target identifiers differ from target-column names"
                        .to_string(),
                );
            }
            let blocks = target_ids
                .iter()
                .map(|target_id| {
                    self.provider
                        .target_block(view_handle, target_id)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            for block in &blocks {
                validate_target_samples(sample_ids, &block.sample_ids)?;
            }
            let mut values = Vec::with_capacity(sample_ids.len() * target_names.len());
            for row in 0..sample_ids.len() {
                for block in &blocks {
                    values.push(finite_number(&block.values[row], "target cell")?);
                }
            }
            values
        };
        Ok(MethodsPlsMatrix {
            values,
            rows: sample_ids.len(),
            cols: target_names.len(),
        })
    }

    pub fn conformal_truth(
        &self,
        replay: &TrainingReplayOutcome,
        binding_id: &str,
    ) -> Result<ConformalCalibrationTruth, String> {
        let output = replay
            .outputs
            .iter()
            .find(|output| output.binding.binding_id == binding_id)
            .ok_or("Core calibration replay has no requested output binding")?;
        if output.binding.target_names != self.provider.target_names() {
            return Err(
                "Core calibration target order differs from the replay output binding".to_string(),
            );
        }
        let [point] = output.predictions.as_slice() else {
            return Err(
                "Core calibration replay requires one terminal point-prediction block".to_string(),
            );
        };
        let handles = self
            .handles
            .lock()
            .map_err(|_| "Core IO provider handle registry is poisoned".to_string())?;
        let predict_views = handles
            .views
            .iter()
            .filter(|slot| slot.phase == Phase::Predict)
            .collect::<Vec<_>>();
        let [slot] = predict_views.as_slice() else {
            return Err(format!(
                "Core calibration replay requires exactly one IO PREDICT view, got {}",
                predict_views.len()
            ));
        };
        let feature = self
            .provider
            .feature_block_f64(slot.handle)
            .map_err(|error| error.to_string())?;
        let sample_ids = feature
            .sample_ids
            .iter()
            .map(|id| dag_ml_core::SampleId::new(id.as_str()).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        if sample_ids != point.sample_ids {
            return Err(
                "Core calibration truth rows differ from replay point-prediction identities"
                    .to_string(),
            );
        }
        let targets = self.project_targets(
            slot.handle,
            &feature.sample_ids,
            self.provider.target_names(),
        )?;
        let values = targets
            .values
            .chunks_exact(targets.cols)
            .map(|row| row.to_vec())
            .collect();
        Ok(ConformalCalibrationTruth { sample_ids, values })
    }
}

impl Drop for DatasetPackageMethodsProvider {
    fn drop(&mut self) {
        self.release_all();
    }
}

/// Closed PLS profiles exposed by the aggregate's canonical request builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalPlsProfile {
    /// PLS over the raw numeric spectrum matrix. Current ABI 2.20 writes N4MM 3;
    /// historical N4MM 1 archives remain readable.
    Raw,
    /// Native SNV → Savitzky-Golay → PLS over raw caller input. Current ABI
    /// 2.20 writes N4MM 3 with an attested pipeline; historical format 2 reads remain.
    SnvSavitzkyGolay,
}

/// Build the aggregate's bounded single-source PLS training request.
///
/// The provider remains authoritative for source, schema, relation and target
/// identities. The request contains one `pls` node with one component, either
/// raw or with the fixed native SNV + SG(window=5, degree=2) pipeline, and a
/// deterministic two-fold sample split. Augmented, excluded, origin-linked,
/// duplicate or undersized cohorts are refused instead of being silently
/// reshaped into this narrow profile.
pub fn canonical_pls_training_request(
    provider: &DatasetPackageMethodsProvider,
    profile: CanonicalPlsProfile,
) -> Result<TrainingRequest, String> {
    base_regression_training_request(provider, profile, false)
}

pub fn base_regression_training_request(
    provider: &DatasetPackageMethodsProvider,
    profile: CanonicalPlsProfile,
    allow_origin_relations: bool,
) -> Result<TrainingRequest, String> {
    let mut sample_ids = Vec::new();
    let mut seen = BTreeSet::new();
    for relation in &provider.relations.records {
        let partitions: Vec<_> = relation
            .tags
            .iter()
            .filter(|tag| tag.starts_with("partition:"))
            .collect();
        if partitions.len() != 1 || partitions[0] != "partition:train" {
            return Err("canonical PLS training requires only train partition rows".into());
        }
        if relation.is_augmented
            || relation.excluded
            || (!allow_origin_relations && relation.origin_sample_id.is_some())
        {
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
    let envelope = provider.external_envelope();
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
        "feature_set_id": provider.provider.feature_set_id(),
        "source_ids": [provider.source_id()],
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
    let node_params = match profile {
        CanonicalPlsProfile::Raw => serde_json::json!({"n_components": 1}),
        CanonicalPlsProfile::SnvSavitzkyGolay => serde_json::json!({
            "n_components": 1,
            "pipeline": {
                "schema_version": 1,
                "pipeline_type": "n4m.snv_savgol_smooth.v1",
                "savgol_window": 5,
                "savgol_poly_degree": 2
            }
        }),
    };
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
                "feature_namespace": provider.source_id(),
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
    let target_names = provider.provider.target_names().to_vec();
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

impl RuntimeDataProvider for DatasetPackageMethodsProvider {
    fn materialize(&self, request: &DataMaterializationRequest) -> dag_ml_core::Result<HandleRef> {
        self.validate_binding(&request.binding)
            .map_err(dag_ml_core::DagMlError::RuntimeValidation)?;
        let io_request = CoordinatorDataMaterializationRequest {
            run_id: request.run_id.to_string(),
            node_id: request.node_id.to_string(),
            input_name: request.input_name.clone(),
            phase: request.phase.as_str().to_string(),
            variant_id: request.variant_id.as_ref().map(ToString::to_string),
            fold_id: request.fold_id.as_ref().map(ToString::to_string),
            request_id: request.binding.request_id.clone(),
            schema_fingerprint: self.provider.envelope().schema_fingerprint.clone(),
            plan_fingerprint: self.provider.envelope().plan_fingerprint.clone(),
            relation_fingerprint: self.provider.envelope().relation_fingerprint.clone(),
            output_representation: self.provider.envelope().plan.output_representation.clone(),
            source_ids: vec![IoSourceId::new(&self.source_id)
                .map_err(|error| dag_ml_core::DagMlError::RuntimeValidation(error.to_string()))?],
            require_relations: request.binding.require_relations,
            predict_cohort: request
                .predict_cohort
                .as_ref()
                .map(|cohort| {
                    serde_json::to_value(cohort)
                        .and_then(serde_json::from_value)
                        .map_err(|error| {
                            dag_ml_core::DagMlError::RuntimeValidation(error.to_string())
                        })
                })
                .transpose()?,
        };
        let record = self
            .provider
            .materialize(&io_request)
            .map_err(|error| dag_ml_core::DagMlError::RuntimeValidation(error.to_string()))?;
        if record.handle.kind != CoordinatorHandleKind::Data {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "IO PackageProvider returned a non-data materialization handle".to_string(),
            ));
        }
        self.handles
            .lock()
            .map_err(|_| {
                dag_ml_core::DagMlError::RuntimeValidation(
                    "Core IO provider handle registry is poisoned".to_string(),
                )
            })?
            .data
            .insert(record.handle.handle);
        Ok(HandleRef {
            handle: record.handle.handle,
            kind: HandleKind::Data,
            owner_controller: ControllerId::new(record.handle.owner_controller)
                .map_err(|error| dag_ml_core::DagMlError::RuntimeValidation(error.to_string()))?,
        })
    }

    fn make_view(&self, request: &DataViewRequest) -> dag_ml_core::Result<HandleRef> {
        self.validate_binding(&request.binding)
            .map_err(dag_ml_core::DagMlError::RuntimeValidation)?;
        let mut handles = self.handles.lock().map_err(|_| {
            dag_ml_core::DagMlError::RuntimeValidation(
                "Core IO provider handle registry is poisoned".to_string(),
            )
        })?;
        if !handles.data.contains(&request.data_handle.handle) {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(format!(
                "Core IO provider does not own data handle {}",
                request.data_handle.handle
            )));
        }
        if request.view.branch_view.is_some() {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "Core DATA-002 Methods training does not support branch/fusion views".to_string(),
            ));
        }
        let view = DataView {
            sample_ids: request
                .view
                .sample_ids
                .as_ref()
                .map(|ids| {
                    ids.iter()
                        .map(|id| IoSampleId::new(id.as_str()).map_err(|error| error.to_string()))
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()
                .map_err(dag_ml_core::DagMlError::RuntimeValidation)?,
            partition: None,
            fold_id: None,
            source_ids: Some(vec![IoSourceId::new(&self.source_id).map_err(|error| {
                dag_ml_core::DagMlError::RuntimeValidation(error.to_string())
            })?]),
            columns: request.view.columns.clone(),
            include_augmented: request.view.include_augmented,
            include_excluded: request.view.include_excluded,
            branch_view: None,
            extra: request.view.extra.clone(),
        };
        let record = self
            .provider
            .make_view(request.data_handle.handle, &view)
            .map_err(|error| dag_ml_core::DagMlError::RuntimeValidation(error.to_string()))?;
        if record.handle.kind != CoordinatorHandleKind::View {
            return Err(dag_ml_core::DagMlError::RuntimeValidation(
                "IO PackageProvider returned a non-view handle".to_string(),
            ));
        }
        handles.views.push(ViewSlot {
            node_id: request.node_id.clone(),
            input_name: request.input_name.clone(),
            phase: request.phase,
            variant_id: request.variant_id.clone(),
            fold_id: request.fold_id.clone(),
            view: request.view.clone(),
            handle: record.handle.handle,
        });
        Ok(HandleRef {
            handle: record.handle.handle,
            kind: HandleKind::DataView,
            owner_controller: ControllerId::new(record.handle.owner_controller)
                .map_err(|error| dag_ml_core::DagMlError::RuntimeValidation(error.to_string()))?,
        })
    }

    fn training_data_identity(
        &self,
        binding: &DataBinding,
    ) -> dag_ml_core::Result<Option<TrainingDataIdentity>> {
        self.validate_binding(binding)
            .map_err(dag_ml_core::DagMlError::RuntimeValidation)?;
        TrainingDataIdentity::from_binding_envelope(binding, &self.external_envelope).map(Some)
    }

    fn coordinator_relations(
        &self,
        binding: &DataBinding,
    ) -> dag_ml_core::Result<Option<SampleRelationSet>> {
        self.validate_binding(binding)
            .map_err(dag_ml_core::DagMlError::RuntimeValidation)?;
        Ok(Some(self.relations.clone()))
    }

    fn methods_pls_capability(&self) -> dag_ml_core::Result<()> {
        Ok(())
    }

    fn preflight_methods_pls(&self, request: &MethodsPlsDataRequest) -> dag_ml_core::Result<()> {
        request.validate()?;
        self.validate_binding(&request.binding)
            .map_err(dag_ml_core::DagMlError::RuntimeValidation)
    }

    fn methods_pls_data(
        &self,
        request: &MethodsPlsDataRequest,
    ) -> dag_ml_core::Result<MethodsPlsData> {
        self.preflight_methods_pls(request)?;
        let fit = self
            .project_dataset(request, &request.fit_view, request.phase != Phase::Predict)
            .map_err(dag_ml_core::DagMlError::RuntimeValidation)?;
        let prediction = request
            .prediction_view
            .as_ref()
            .map(|view| self.project_dataset(request, view, true))
            .transpose()
            .map_err(dag_ml_core::DagMlError::RuntimeValidation)?;
        let data = MethodsPlsData { fit, prediction };
        data.validate_for(request)?;
        Ok(data)
    }
}

pub fn convert_relations(
    relations: &dag_ml_data::CoordinatorRelationSet,
) -> Result<SampleRelationSet, String> {
    let records = relations
        .records
        .iter()
        .map(|record| {
            Ok(SampleRelation {
                unit_level: EntityUnitLevel::Observation,
                unit_id: None,
                observation_id: dag_ml_core::ObservationId::new(record.observation_id.as_str())
                    .map_err(|error| error.to_string())?,
                sample_id: dag_ml_core::SampleId::new(record.sample_id.as_str())
                    .map_err(|error| error.to_string())?,
                source_id: record.source_id.as_ref().map(ToString::to_string),
                rep_id: None,
                target_id: record
                    .target_id
                    .as_ref()
                    .map(|id| dag_ml_core::TargetId::new(id.as_str()))
                    .transpose()
                    .map_err(|error| error.to_string())?,
                group_id: record
                    .group_id
                    .as_ref()
                    .map(|id| dag_ml_core::GroupId::new(id.as_str()))
                    .transpose()
                    .map_err(|error| error.to_string())?,
                origin_sample_id: record
                    .origin_sample_id
                    .as_ref()
                    .map(|id| dag_ml_core::SampleId::new(id.as_str()))
                    .transpose()
                    .map_err(|error| error.to_string())?,
                derived_unit_id: None,
                component_observation_ids: Vec::new(),
                sample_influence_weight: None,
                quality_flag: None,
                is_augmented: record.is_augmented,
                excluded: record.excluded,
                metadata: record.metadata.clone(),
                tags: record.tags.clone(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let converted = SampleRelationSet { records };
    converted.validate().map_err(|error| error.to_string())?;
    Ok(converted)
}

fn validate_target_samples(expected: &[IoSampleId], actual: &[IoSampleId]) -> Result<(), String> {
    if expected != actual {
        return Err(
            "Core IO provider target rows differ from feature sample identities".to_string(),
        );
    }
    Ok(())
}

fn finite_number(value: &serde_json::Value, label: &str) -> Result<f64, String> {
    let number = value
        .as_f64()
        .ok_or_else(|| format!("Core IO provider {label} is not numeric"))?;
    if !number.is_finite() {
        return Err(format!("Core IO provider {label} is non-finite"));
    }
    Ok(number)
}
