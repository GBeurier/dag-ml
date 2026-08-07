//! Public training replay contracts.
//!
//! This module owns the strict, portable `TrainingReplayRequest` and
//! `TrainingReplayOutcome` contracts introduced before the attached replay
//! runtime. The low-level `ReplayPhaseRequest` in `bundle` remains the internal
//! phase API; these types are the public training-owned replay surface.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::bundle::{ExecutionBundle, ReplayPhaseRequest};
use crate::campaign::stable_json_fingerprint;
use crate::canonical::parse_typed_json;
use crate::data::ExternalDataPlanEnvelope;
use crate::error::{DagMlError, Result};
use crate::ids::{ArtifactId, BundleId, RunId};
use crate::phase::Phase;
use crate::plan::ExecutionPlan;
use crate::runtime::{
    ArtifactMaterializationRequest, BundleReplayExecution, ExplanationBlock, HandleRef,
    LineageRecord, RunContext, RuntimeArtifactStore, RuntimeControllerRegistry,
    RuntimeDataProvider, SequentialScheduler,
};
use crate::training::{
    LoadedPredictor, PortablePredictorPackage, TrainingDataIdentity, TrainingOutcomeRef,
};
use crate::training_runtime::{
    BoundTrainingOutput, TrainingOutcome, BOUND_TRAINING_OUTPUT_SCHEMA_VERSION,
};

pub const TRAINING_REPLAY_REQUEST_SCHEMA_VERSION: u32 = 1;
pub const TRAINING_REPLAY_OUTCOME_SCHEMA_VERSION: u32 = 1;

pub struct AttachedTrainingReplayInput<'a> {
    pub source: &'a TrainingOutcome,
    pub request: &'a TrainingReplayRequest,
    pub outcome_id: String,
    pub run_id: RunId,
    pub controllers: &'a RuntimeControllerRegistry,
    pub data_provider: &'a dyn RuntimeDataProvider,
    pub artifact_store: &'a dyn RuntimeArtifactStore,
    pub data_envelopes: &'a BTreeMap<String, ExternalDataPlanEnvelope>,
    pub warnings: Vec<String>,
    pub diagnostics: BTreeMap<String, serde_json::Value>,
}

pub struct LoadedPredictorReplayInput<'a> {
    pub predictor: &'a LoadedPredictor<HandleRef>,
    pub request: &'a TrainingReplayRequest,
    pub outcome_id: String,
    pub run_id: RunId,
    pub controllers: &'a RuntimeControllerRegistry,
    pub data_provider: &'a dyn RuntimeDataProvider,
    pub data_envelopes: &'a BTreeMap<String, ExternalDataPlanEnvelope>,
    pub warnings: Vec<String>,
    pub diagnostics: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingReplayRequest {
    pub schema_version: u32,
    pub request_id: String,
    pub source_outcome_fingerprint: String,
    pub phase: Phase,
    pub data_envelope_keys: Vec<String>,
    pub output_binding_ids: Vec<String>,
    pub request_fingerprint: String,
}

impl TrainingReplayRequest {
    pub fn from_json(json: &str) -> Result<Self> {
        let raw_fingerprint = strict_tcv1_fingerprint_without(
            json,
            "request_fingerprint",
            "training replay request",
        )?;
        let request: Self = serde_json::from_str(json)?;
        if request.request_fingerprint != raw_fingerprint {
            return contract_error(
                "training replay request fingerprint does not match original TCV1 JSON",
            );
        }
        request.validate()?;
        Ok(request)
    }

    pub fn compute_fingerprint(&self) -> Result<String> {
        tcv1_fingerprint_without(self, "request_fingerprint", "training replay request")
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != TRAINING_REPLAY_REQUEST_SCHEMA_VERSION {
            return unsupported_version(
                "training replay request",
                self.schema_version,
                TRAINING_REPLAY_REQUEST_SCHEMA_VERSION,
            );
        }
        validate_identifier("training replay request_id", &self.request_id)?;
        validate_sha256(
            "training replay source outcome",
            &self.source_outcome_fingerprint,
        )?;
        validate_replay_phase(self.phase)?;
        validate_sorted_unique_text(
            "training replay data_envelope_keys",
            &self.data_envelope_keys,
            true,
        )?;
        validate_sorted_unique_identifiers(
            "training replay output_binding_ids",
            &self.output_binding_ids,
            true,
        )?;
        validate_sha256("training replay request", &self.request_fingerprint)?;
        if self.request_fingerprint != self.compute_fingerprint()? {
            return contract_error(
                "training replay request fingerprint does not match TCV1 content",
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingReplayOutcome {
    pub schema_version: u32,
    pub outcome_id: String,
    pub run_id: RunId,
    pub source_training_outcome: TrainingOutcomeRef,
    pub replay_request_id: String,
    pub replay_request_fingerprint: String,
    pub input_data_identities: Vec<TrainingDataIdentity>,
    pub bundle_id: BundleId,
    pub plan_id: String,
    pub phase: Phase,
    pub result_count: usize,
    pub lineage_record_count: usize,
    pub prediction_block_count: usize,
    pub observation_prediction_block_count: usize,
    pub aggregated_prediction_block_count: usize,
    pub explanation_block_count: usize,
    pub controller_count: usize,
    pub prediction_cache_store: bool,
    pub outputs: Vec<BoundTrainingOutput>,
    pub explanations: Vec<ExplanationBlock>,
    pub lineage: Vec<LineageRecord>,
    pub warnings: Vec<String>,
    pub diagnostics: BTreeMap<String, serde_json::Value>,
    pub outcome_fingerprint: String,
}

impl TrainingReplayOutcome {
    pub fn from_json(json: &str) -> Result<Self> {
        let raw_fingerprint = strict_tcv1_fingerprint_without(
            json,
            "outcome_fingerprint",
            "training replay outcome",
        )?;
        let outcome: Self = serde_json::from_str(json)?;
        if outcome.outcome_fingerprint != raw_fingerprint {
            return contract_error(
                "training replay outcome fingerprint does not match original TCV1 JSON",
            );
        }
        outcome.validate()?;
        Ok(outcome)
    }

    pub fn compute_fingerprint(&self) -> Result<String> {
        tcv1_fingerprint_without(self, "outcome_fingerprint", "training replay outcome")
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != TRAINING_REPLAY_OUTCOME_SCHEMA_VERSION {
            return unsupported_version(
                "training replay outcome",
                self.schema_version,
                TRAINING_REPLAY_OUTCOME_SCHEMA_VERSION,
            );
        }
        validate_identifier("training replay outcome_id", &self.outcome_id)?;
        validate_identifier("training replay request_id", &self.replay_request_id)?;
        validate_sha256("training replay request", &self.replay_request_fingerprint)?;
        validate_non_empty("training replay plan_id", &self.plan_id)?;
        validate_replay_phase(self.phase)?;
        self.source_training_outcome.validate()?;
        for identity in &self.input_data_identities {
            identity.validate()?;
        }
        validate_sorted_unique_keys(
            "training replay input_data_identities",
            self.input_data_identities
                .iter()
                .map(|identity| identity.requirement_key.as_str()),
            true,
        )?;
        if self.prediction_cache_store {
            return contract_error("training replay outcome cannot persist a prediction cache");
        }
        validate_sorted_unique_text("training replay warnings", &self.warnings, false)?;
        validate_diagnostics(&self.diagnostics)?;
        validate_output_order_and_version(&self.outputs)?;
        for output in &self.outputs {
            validate_replay_bound_output_blocks(output)?;
        }
        for explanation in &self.explanations {
            explanation.validate()?;
            validate_optional_port(
                "training replay explanation producer_port",
                &explanation.producer_port,
            )?;
        }
        for record in &self.lineage {
            record.validate()?;
        }
        match self.phase {
            Phase::Predict if self.outputs.is_empty() => {
                return contract_error("training replay PREDICT requires at least one output");
            }
            Phase::Predict if !self.explanations.is_empty() => {
                return contract_error("training replay PREDICT cannot emit explanations");
            }
            Phase::Explain if self.explanations.is_empty() => {
                return contract_error("training replay EXPLAIN requires at least one explanation");
            }
            _ => {}
        }
        self.validate_counters()?;
        validate_sha256("training replay outcome", &self.outcome_fingerprint)?;
        if self.outcome_fingerprint != self.compute_fingerprint()? {
            return contract_error(
                "training replay outcome fingerprint does not match TCV1 content",
            );
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        source: &TrainingOutcome,
        request: &TrainingReplayRequest,
    ) -> Result<()> {
        self.validate()?;
        source.validate()?;
        request.validate()?;
        if request.source_outcome_fingerprint != source.outcome_fingerprint {
            return contract_error("training replay request does not target source outcome");
        }
        if !source.replayable_phases.contains(&request.phase) {
            return contract_error("training replay phase is not replayable by source outcome");
        }
        if self.source_training_outcome != source.to_reference()? {
            return contract_error(
                "training replay outcome source reference does not match source outcome",
            );
        }
        if self.replay_request_id != request.request_id {
            return contract_error(
                "training replay outcome request id does not match ReplayRequest",
            );
        }
        if self.replay_request_fingerprint != request.request_fingerprint {
            return contract_error(
                "training replay outcome request fingerprint does not match ReplayRequest",
            );
        }
        if self.phase != request.phase {
            return contract_error("training replay outcome phase does not match ReplayRequest");
        }
        if self.bundle_id != source.execution_bundle.bundle_id {
            return contract_error("training replay outcome bundle does not match source outcome");
        }
        if self.plan_id != source.effective_plan.id {
            return contract_error("training replay outcome plan does not match source outcome");
        }
        let identity_keys = self
            .input_data_identities
            .iter()
            .map(|identity| identity.requirement_key.clone())
            .collect::<Vec<_>>();
        if identity_keys != request.data_envelope_keys {
            return contract_error(
                "training replay outcome identities do not exactly cover ReplayRequest envelopes",
            );
        }
        let source_bindings = source
            .outputs
            .iter()
            .map(|output| (output.binding.binding_id.as_str(), &output.binding))
            .collect::<BTreeMap<_, _>>();
        for binding_id in &request.output_binding_ids {
            if !source_bindings.contains_key(binding_id.as_str()) {
                return contract_error(
                    "training replay request references absent source output binding",
                );
            }
        }
        let emitted_binding_ids = self
            .outputs
            .iter()
            .map(|output| output.binding.binding_id.clone())
            .collect::<Vec<_>>();
        if self.phase == Phase::Predict && emitted_binding_ids != request.output_binding_ids {
            return contract_error(
                "training replay PREDICT outputs do not exactly cover ReplayRequest bindings",
            );
        }
        if self.phase == Phase::Explain
            && !emitted_binding_ids
                .iter()
                .all(|binding_id| request.output_binding_ids.contains(binding_id))
        {
            return contract_error(
                "training replay EXPLAIN outputs include a binding outside ReplayRequest",
            );
        }
        for output in &self.outputs {
            let Some(source_binding) = source_bindings.get(output.binding.binding_id.as_str())
            else {
                return contract_error(
                    "training replay output binding is absent from source outcome",
                );
            };
            if &output.binding != *source_binding {
                return contract_error(
                    "training replay output binding does not match source outcome binding",
                );
            }
            output.validate(&source.effective_plan)?;
        }
        Ok(())
    }

    pub fn validate_against_package(
        &self,
        package: &PortablePredictorPackage,
        request: &TrainingReplayRequest,
    ) -> Result<()> {
        self.validate()?;
        package.validate()?;
        request.validate()?;
        validate_replay_phase(request.phase)?;
        if request.source_outcome_fingerprint != package.training_outcome.outcome_fingerprint {
            return contract_error(
                "training replay request does not target package source outcome",
            );
        }
        if self.source_training_outcome != package.training_outcome {
            return contract_error(
                "training replay outcome source reference does not match package source outcome",
            );
        }
        if self.replay_request_id != request.request_id {
            return contract_error(
                "training replay outcome request id does not match ReplayRequest",
            );
        }
        if self.replay_request_fingerprint != request.request_fingerprint {
            return contract_error(
                "training replay outcome request fingerprint does not match ReplayRequest",
            );
        }
        if self.phase != request.phase {
            return contract_error("training replay outcome phase does not match ReplayRequest");
        }
        if self.bundle_id != package.execution_bundle.bundle_id {
            return contract_error("training replay outcome bundle does not match package");
        }
        if self.plan_id != package.effective_plan.id {
            return contract_error("training replay outcome plan does not match package");
        }
        let identity_keys = self
            .input_data_identities
            .iter()
            .map(|identity| identity.requirement_key.clone())
            .collect::<Vec<_>>();
        if identity_keys != request.data_envelope_keys {
            return contract_error(
                "training replay outcome identities do not exactly cover ReplayRequest envelopes",
            );
        }
        let package_bindings = package
            .output_bindings
            .iter()
            .map(|binding| (binding.binding_id.as_str(), binding))
            .collect::<BTreeMap<_, _>>();
        for binding_id in &request.output_binding_ids {
            if !package_bindings.contains_key(binding_id.as_str()) {
                return contract_error(
                    "training replay request references absent package output binding",
                );
            }
        }
        let emitted_binding_ids = self
            .outputs
            .iter()
            .map(|output| output.binding.binding_id.clone())
            .collect::<Vec<_>>();
        if self.phase == Phase::Predict && emitted_binding_ids != request.output_binding_ids {
            return contract_error(
                "training replay PREDICT outputs do not exactly cover ReplayRequest bindings",
            );
        }
        if self.phase == Phase::Explain
            && !emitted_binding_ids
                .iter()
                .all(|binding_id| request.output_binding_ids.contains(binding_id))
        {
            return contract_error(
                "training replay EXPLAIN outputs include a binding outside ReplayRequest",
            );
        }
        for output in &self.outputs {
            let Some(package_binding) = package_bindings.get(output.binding.binding_id.as_str())
            else {
                return contract_error("training replay output binding is absent from package");
            };
            if &output.binding != *package_binding {
                return contract_error(
                    "training replay output binding does not match package binding",
                );
            }
            output.validate(&package.effective_plan)?;
        }
        Ok(())
    }

    fn validate_counters(&self) -> Result<()> {
        require_count(
            "training replay result_count",
            self.result_count,
            self.lineage.len(),
        )?;
        require_count(
            "training replay lineage_record_count",
            self.lineage_record_count,
            self.lineage.len(),
        )?;
        require_count(
            "training replay prediction_block_count",
            self.prediction_block_count,
            self.outputs
                .iter()
                .map(|output| output.predictions.len())
                .sum(),
        )?;
        require_count(
            "training replay observation_prediction_block_count",
            self.observation_prediction_block_count,
            self.outputs
                .iter()
                .map(|output| output.observation_predictions.len())
                .sum(),
        )?;
        require_count(
            "training replay aggregated_prediction_block_count",
            self.aggregated_prediction_block_count,
            self.outputs
                .iter()
                .map(|output| output.aggregated_predictions.len())
                .sum(),
        )?;
        require_count(
            "training replay explanation_block_count",
            self.explanation_block_count,
            self.explanations.len(),
        )?;
        let controller_count = self
            .lineage
            .iter()
            .map(|record| record.controller_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        require_count(
            "training replay controller_count",
            self.controller_count,
            controller_count,
        )?;
        Ok(())
    }
}

struct LoadedPredictorArtifactStore<'a> {
    predictor: &'a LoadedPredictor<HandleRef>,
    records: BTreeMap<ArtifactId, crate::bundle::RefitArtifactRecord>,
}

impl<'a> LoadedPredictorArtifactStore<'a> {
    fn new(predictor: &'a LoadedPredictor<HandleRef>) -> Result<Self> {
        predictor.package().validate()?;
        let records = predictor
            .package()
            .execution_bundle
            .refit_artifacts
            .iter()
            .map(|record| {
                record.validate()?;
                Ok((record.artifact.id.clone(), record.clone()))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self { predictor, records })
    }
}

impl RuntimeArtifactStore for LoadedPredictorArtifactStore<'_> {
    fn materialize(&self, request: &ArtifactMaterializationRequest) -> Result<HandleRef> {
        let record = self.records.get(&request.artifact.id).ok_or_else(|| {
            DagMlError::RuntimeValidation(format!(
                "loaded predictor is missing refit artifact `{}` for bundle `{}`",
                request.artifact.id, request.bundle_id
            ))
        })?;
        if record.node_id != request.node_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` is registered for node `{}` but requested for `{}`",
                request.artifact.id, record.node_id, request.node_id
            )));
        }
        if record.controller_id != request.controller_id {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` is registered for controller `{}` but requested for `{}`",
                request.artifact.id, record.controller_id, request.controller_id
            )));
        }
        if record.artifact != request.artifact {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` metadata does not match package bundle record",
                request.artifact.id
            )));
        }
        if record.params_fingerprint != request.params_fingerprint {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` params fingerprint does not match package bundle record",
                request.artifact.id
            )));
        }
        if record.training_loss_fingerprint != request.training_loss_fingerprint {
            return Err(DagMlError::RuntimeValidation(format!(
                "artifact `{}` training loss fingerprint does not match package bundle record",
                request.artifact.id
            )));
        }
        let handle = self
            .predictor
            .artifact(&request.artifact.id)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "loaded predictor has no process-local handle for `{}`",
                    request.artifact.id
                ))
            })?;
        Ok(handle.clone())
    }
}

pub fn execute_attached_training_replay(
    input: AttachedTrainingReplayInput<'_>,
) -> Result<TrainingReplayOutcome> {
    input.source.validate()?;
    input.request.validate()?;
    validate_sorted_unique_text("training replay execution warnings", &input.warnings, false)?;
    validate_diagnostics(&input.diagnostics)?;
    if input.request.source_outcome_fingerprint != input.source.outcome_fingerprint {
        return contract_error("training replay request does not target source outcome");
    }
    if !input
        .source
        .replayable_phases
        .contains(&input.request.phase)
    {
        return contract_error("training replay phase is not replayable by source outcome");
    }
    for node_plan in input.source.effective_plan.node_plans.values() {
        if input.controllers.get(&node_plan.controller_id).is_none() {
            return Err(DagMlError::RuntimeValidation(format!(
                "attached training replay controller `{}` for node `{}` is not registered",
                node_plan.controller_id, node_plan.node_id
            )));
        }
    }

    let input_data_identities = replay_input_data_identities(
        &input.source.execution_bundle,
        input.request,
        input.data_envelopes,
    )?;
    let (replay_plan, replay_bundle) = replay_plan_and_bundle_for_current_cohort(
        &input.source.effective_plan,
        &input.source.execution_bundle,
        input.request,
        input.data_envelopes,
    )?;
    let phase_request = ReplayPhaseRequest {
        bundle_id: replay_bundle.bundle_id.clone(),
        phase: input.request.phase,
        data_envelope_keys: input.request.data_envelope_keys.clone(),
    };
    let mut ctx = RunContext::new(input.run_id.clone(), None);
    let results = SequentialScheduler.execute_bundle_replay(
        BundleReplayExecution {
            plan: &replay_plan,
            bundle: &replay_bundle,
            replay_request: &phase_request,
            prediction_cache_store: None,
            controllers: input.controllers,
            data_provider: input.data_provider,
            artifact_store: input.artifact_store,
            data_envelopes: input.data_envelopes,
        },
        &mut ctx,
    )?;
    if results
        .iter()
        .any(|result| !result.artifacts.is_empty() || !result.artifact_handles.is_empty())
    {
        return contract_error("attached training replay PREDICT/EXPLAIN cannot emit artifacts");
    }

    let outputs = bind_attached_replay_outputs(input.source, input.request, &results)?;
    let explanations = bind_attached_replay_explanations(input.request, &results)?;
    let mut lineage = ctx.lineage.records().cloned().collect::<Vec<_>>();
    for record in &mut lineage {
        record.input_lineage.sort();
        record
            .artifact_refs
            .sort_by(|left, right| left.id.cmp(&right.id));
    }
    lineage.sort_by(|left, right| left.record_id.cmp(&right.record_id));

    let mut outcome = TrainingReplayOutcome {
        schema_version: TRAINING_REPLAY_OUTCOME_SCHEMA_VERSION,
        outcome_id: input.outcome_id,
        run_id: input.run_id,
        source_training_outcome: input.source.to_reference()?,
        replay_request_id: input.request.request_id.clone(),
        replay_request_fingerprint: input.request.request_fingerprint.clone(),
        input_data_identities,
        bundle_id: input.source.execution_bundle.bundle_id.clone(),
        plan_id: input.source.effective_plan.id.clone(),
        phase: input.request.phase,
        result_count: lineage.len(),
        lineage_record_count: lineage.len(),
        prediction_block_count: outputs.iter().map(|output| output.predictions.len()).sum(),
        observation_prediction_block_count: outputs
            .iter()
            .map(|output| output.observation_predictions.len())
            .sum(),
        aggregated_prediction_block_count: outputs
            .iter()
            .map(|output| output.aggregated_predictions.len())
            .sum(),
        explanation_block_count: explanations.len(),
        controller_count: lineage
            .iter()
            .map(|record| record.controller_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        prediction_cache_store: false,
        outputs,
        explanations,
        lineage,
        warnings: input.warnings,
        diagnostics: input.diagnostics,
        outcome_fingerprint: zero_fingerprint(),
    };
    outcome.outcome_fingerprint = outcome.compute_fingerprint()?;
    outcome.validate_against(input.source, input.request)?;
    Ok(outcome)
}

pub fn execute_loaded_predictor_replay(
    input: LoadedPredictorReplayInput<'_>,
) -> Result<TrainingReplayOutcome> {
    let package = input.predictor.package();
    package.validate()?;
    input.request.validate()?;
    validate_sorted_unique_text("training replay execution warnings", &input.warnings, false)?;
    validate_diagnostics(&input.diagnostics)?;
    validate_replay_phase(input.request.phase)?;
    if input.request.source_outcome_fingerprint != package.training_outcome.outcome_fingerprint {
        return contract_error("training replay request does not target package source outcome");
    }
    for node_plan in package.effective_plan.node_plans.values() {
        if input.controllers.get(&node_plan.controller_id).is_none() {
            return Err(DagMlError::RuntimeValidation(format!(
                "loaded predictor replay controller `{}` for node `{}` is not registered",
                node_plan.controller_id, node_plan.node_id
            )));
        }
    }

    let input_data_identities = replay_input_data_identities(
        &package.execution_bundle,
        input.request,
        input.data_envelopes,
    )?;
    let (replay_plan, replay_bundle) = replay_plan_and_bundle_for_current_cohort(
        &package.effective_plan,
        &package.execution_bundle,
        input.request,
        input.data_envelopes,
    )?;
    let phase_request = ReplayPhaseRequest {
        bundle_id: replay_bundle.bundle_id.clone(),
        phase: input.request.phase,
        data_envelope_keys: input.request.data_envelope_keys.clone(),
    };
    let artifact_store = LoadedPredictorArtifactStore::new(input.predictor)?;
    let mut ctx = RunContext::new(input.run_id.clone(), None);
    let results = SequentialScheduler.execute_bundle_replay(
        BundleReplayExecution {
            plan: &replay_plan,
            bundle: &replay_bundle,
            replay_request: &phase_request,
            prediction_cache_store: None,
            controllers: input.controllers,
            data_provider: input.data_provider,
            artifact_store: &artifact_store,
            data_envelopes: input.data_envelopes,
        },
        &mut ctx,
    )?;
    if results
        .iter()
        .any(|result| !result.artifacts.is_empty() || !result.artifact_handles.is_empty())
    {
        return contract_error("loaded predictor replay PREDICT/EXPLAIN cannot emit artifacts");
    }

    let outputs = bind_package_replay_outputs(package, input.request, &results)?;
    let explanations = bind_attached_replay_explanations(input.request, &results)?;
    let mut lineage = ctx.lineage.records().cloned().collect::<Vec<_>>();
    for record in &mut lineage {
        record.input_lineage.sort();
        record
            .artifact_refs
            .sort_by(|left, right| left.id.cmp(&right.id));
    }
    lineage.sort_by(|left, right| left.record_id.cmp(&right.record_id));

    let mut outcome = TrainingReplayOutcome {
        schema_version: TRAINING_REPLAY_OUTCOME_SCHEMA_VERSION,
        outcome_id: input.outcome_id,
        run_id: input.run_id,
        source_training_outcome: package.training_outcome.clone(),
        replay_request_id: input.request.request_id.clone(),
        replay_request_fingerprint: input.request.request_fingerprint.clone(),
        input_data_identities,
        bundle_id: package.execution_bundle.bundle_id.clone(),
        plan_id: package.effective_plan.id.clone(),
        phase: input.request.phase,
        result_count: lineage.len(),
        lineage_record_count: lineage.len(),
        prediction_block_count: outputs.iter().map(|output| output.predictions.len()).sum(),
        observation_prediction_block_count: outputs
            .iter()
            .map(|output| output.observation_predictions.len())
            .sum(),
        aggregated_prediction_block_count: outputs
            .iter()
            .map(|output| output.aggregated_predictions.len())
            .sum(),
        explanation_block_count: explanations.len(),
        controller_count: lineage
            .iter()
            .map(|record| record.controller_id.as_str())
            .collect::<BTreeSet<_>>()
            .len(),
        prediction_cache_store: false,
        outputs,
        explanations,
        lineage,
        warnings: input.warnings,
        diagnostics: input.diagnostics,
        outcome_fingerprint: zero_fingerprint(),
    };
    outcome.outcome_fingerprint = outcome.compute_fingerprint()?;
    outcome.validate_against_package(package, input.request)?;
    Ok(outcome)
}

fn replay_input_data_identities(
    bundle: &ExecutionBundle,
    request: &TrainingReplayRequest,
    envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
) -> Result<Vec<TrainingDataIdentity>> {
    request
        .data_envelope_keys
        .iter()
        .map(|key| {
            let requirement = bundle
                .data_requirements
                .iter()
                .find(|requirement| requirement.key() == *key)
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "training replay request references unknown data envelope key `{key}`"
                    ))
                })?;
            let envelope = envelopes.get(key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "training replay is missing external data envelope for `{key}`"
                ))
            })?;
            envelope.validate()?;
            if requirement.schema_fingerprint != envelope.schema_fingerprint
                || requirement.plan_fingerprint != envelope.plan_fingerprint
            {
                return Err(DagMlError::RuntimeValidation(format!(
                    "training replay envelope for `{key}` changes schema or representation plan"
                )));
            }
            let relation_fingerprint = envelope.relation_fingerprint.clone().ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "training replay envelope for `{key}` requires a relation fingerprint"
                ))
            })?;
            let data_content_fingerprint =
                envelope.data_content_fingerprint.clone().ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "training replay envelope for `{key}` requires a data content fingerprint"
                    ))
                })?;
            let target_content_fingerprint =
                envelope.target_content_fingerprint.clone().ok_or_else(|| {
                    DagMlError::RuntimeValidation(format!(
                        "training replay envelope for `{key}` requires a target content fingerprint"
                    ))
                })?;
            let mut identity = TrainingDataIdentity {
                requirement_key: key.clone(),
                schema_fingerprint: envelope.schema_fingerprint.clone(),
                plan_fingerprint: envelope.plan_fingerprint.clone(),
                relation_fingerprint,
                data_content_fingerprint,
                target_content_fingerprint,
                identity_fingerprint: zero_fingerprint(),
            };
            identity.identity_fingerprint = identity.compute_fingerprint()?;
            identity.validate()?;
            Ok(identity)
        })
        .collect()
}

fn replay_plan_and_bundle_for_current_cohort(
    plan: &ExecutionPlan,
    bundle: &ExecutionBundle,
    request: &TrainingReplayRequest,
    envelopes: &BTreeMap<String, ExternalDataPlanEnvelope>,
) -> Result<(ExecutionPlan, ExecutionBundle)> {
    let mut replay_plan = plan.clone();
    let mut replay_bundle = bundle.clone();
    for requirement in &mut replay_bundle.data_requirements {
        let key = requirement.key();
        if request.data_envelope_keys.contains(&key) {
            let envelope = envelopes.get(&key).ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "training replay is missing external data envelope for `{key}`"
                ))
            })?;
            requirement.relation_fingerprint = envelope.relation_fingerprint.clone();
            for bindings in replay_plan.campaign.data_bindings.values_mut() {
                for binding in bindings {
                    if crate::data::data_binding_requirement_key(
                        &binding.node_id,
                        &binding.input_name,
                    ) == key
                    {
                        binding.relation_fingerprint = envelope.relation_fingerprint.clone();
                    }
                }
            }
            for node_plan in replay_plan.node_plans.values_mut() {
                for binding in &mut node_plan.data_bindings {
                    if crate::data::data_binding_requirement_key(
                        &binding.node_id,
                        &binding.input_name,
                    ) == key
                    {
                        binding.relation_fingerprint = envelope.relation_fingerprint.clone();
                    }
                }
            }
        }
    }
    replay_plan.campaign_fingerprint = stable_json_fingerprint(&replay_plan.campaign)?;
    replay_bundle.campaign_fingerprint = replay_plan.campaign_fingerprint.clone();
    Ok((replay_plan, replay_bundle))
}

fn bind_attached_replay_outputs(
    source: &TrainingOutcome,
    request: &TrainingReplayRequest,
    results: &[crate::runtime::NodeResult],
) -> Result<Vec<BoundTrainingOutput>> {
    let mut outputs = Vec::new();
    for binding_id in &request.output_binding_ids {
        let source_output = source
            .outputs
            .iter()
            .find(|output| output.binding.binding_id == *binding_id)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "training replay request references absent binding `{binding_id}`"
                ))
            })?;
        let binding = source_output.binding.clone();
        let mut output = BoundTrainingOutput {
            schema_version: Some(BOUND_TRAINING_OUTPUT_SCHEMA_VERSION),
            binding: binding.clone(),
            predictions: Vec::new(),
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
        };
        for result in results {
            output.predictions.extend(
                result
                    .predictions
                    .iter()
                    .filter(|block| {
                        block.producer_node == binding.node_id
                            && block.producer_port.as_deref() == Some(binding.port_name.as_str())
                            && block.partition == crate::oof::PredictionPartition::Final
                            && block.fold_id.is_none()
                    })
                    .cloned(),
            );
            output.observation_predictions.extend(
                result
                    .observation_predictions
                    .iter()
                    .filter(|block| {
                        block.producer_node == binding.node_id
                            && block.producer_port.as_deref() == Some(binding.port_name.as_str())
                            && block.partition == crate::oof::PredictionPartition::Final
                            && block.fold_id.is_none()
                    })
                    .cloned(),
            );
            output.aggregated_predictions.extend(
                result
                    .aggregated_predictions
                    .iter()
                    .filter(|block| {
                        block.producer_node == binding.node_id
                            && block.producer_port.as_deref() == Some(binding.port_name.as_str())
                            && block.partition == crate::oof::PredictionPartition::Final
                            && block.fold_id.is_none()
                    })
                    .cloned(),
            );
        }
        if !output.predictions.is_empty()
            || !output.observation_predictions.is_empty()
            || !output.aggregated_predictions.is_empty()
        {
            output.validate(&source.effective_plan)?;
            outputs.push(output);
        }
    }
    outputs.sort_by(|left, right| left.binding.binding_id.cmp(&right.binding.binding_id));
    Ok(outputs)
}

fn bind_package_replay_outputs(
    package: &PortablePredictorPackage,
    request: &TrainingReplayRequest,
    results: &[crate::runtime::NodeResult],
) -> Result<Vec<BoundTrainingOutput>> {
    let mut outputs = Vec::new();
    for binding_id in &request.output_binding_ids {
        let binding = package
            .output_bindings
            .iter()
            .find(|binding| binding.binding_id == *binding_id)
            .ok_or_else(|| {
                DagMlError::RuntimeValidation(format!(
                    "training replay request references absent package binding `{binding_id}`"
                ))
            })?
            .clone();
        let mut output = BoundTrainingOutput {
            schema_version: Some(BOUND_TRAINING_OUTPUT_SCHEMA_VERSION),
            binding: binding.clone(),
            predictions: Vec::new(),
            observation_predictions: Vec::new(),
            aggregated_predictions: Vec::new(),
        };
        for result in results {
            output.predictions.extend(
                result
                    .predictions
                    .iter()
                    .filter(|block| {
                        block.producer_node == binding.node_id
                            && block.producer_port.as_deref() == Some(binding.port_name.as_str())
                            && block.partition == crate::oof::PredictionPartition::Final
                            && block.fold_id.is_none()
                    })
                    .cloned(),
            );
            output.observation_predictions.extend(
                result
                    .observation_predictions
                    .iter()
                    .filter(|block| {
                        block.producer_node == binding.node_id
                            && block.producer_port.as_deref() == Some(binding.port_name.as_str())
                            && block.partition == crate::oof::PredictionPartition::Final
                            && block.fold_id.is_none()
                    })
                    .cloned(),
            );
            output.aggregated_predictions.extend(
                result
                    .aggregated_predictions
                    .iter()
                    .filter(|block| {
                        block.producer_node == binding.node_id
                            && block.producer_port.as_deref() == Some(binding.port_name.as_str())
                            && block.partition == crate::oof::PredictionPartition::Final
                            && block.fold_id.is_none()
                    })
                    .cloned(),
            );
        }
        if !output.predictions.is_empty()
            || !output.observation_predictions.is_empty()
            || !output.aggregated_predictions.is_empty()
        {
            output.validate(&package.effective_plan)?;
            outputs.push(output);
        }
    }
    outputs.sort_by(|left, right| left.binding.binding_id.cmp(&right.binding.binding_id));
    Ok(outputs)
}

fn bind_attached_replay_explanations(
    request: &TrainingReplayRequest,
    results: &[crate::runtime::NodeResult],
) -> Result<Vec<ExplanationBlock>> {
    if request.phase != Phase::Explain {
        return Ok(Vec::new());
    }
    let mut explanations = results
        .iter()
        .flat_map(|result| result.explanations.iter().cloned())
        .filter(|block| block.producer_port.is_some())
        .collect::<Vec<_>>();
    explanations.sort_by(|left, right| {
        (
            left.producer_node.as_str(),
            left.producer_port.as_deref().unwrap_or_default(),
            left.method.as_str(),
            left.target_name.as_deref().unwrap_or_default(),
        )
            .cmp(&(
                right.producer_node.as_str(),
                right.producer_port.as_deref().unwrap_or_default(),
                right.method.as_str(),
                right.target_name.as_deref().unwrap_or_default(),
            ))
    });
    Ok(explanations)
}

fn validate_output_order_and_version(outputs: &[BoundTrainingOutput]) -> Result<()> {
    let mut previous: Option<&str> = None;
    for output in outputs {
        match output.schema_version {
            Some(BOUND_TRAINING_OUTPUT_SCHEMA_VERSION) => {}
            Some(version) => {
                return contract_error(format!(
                    "training replay output schema_version {version} is unsupported; current {BOUND_TRAINING_OUTPUT_SCHEMA_VERSION}"
                ));
            }
            None => {
                return contract_error(
                    "training replay output requires bound_training_output schema_version",
                );
            }
        }
        let binding_id = output.binding.binding_id.as_str();
        if previous.is_some_and(|previous| previous >= binding_id) {
            return contract_error("training replay outputs must be strictly sorted by binding_id");
        }
        previous = Some(binding_id);
    }
    Ok(())
}

fn validate_replay_bound_output_blocks(output: &BoundTrainingOutput) -> Result<()> {
    for block in &output.predictions {
        validate_optional_port(
            "training replay prediction producer_port",
            &block.producer_port,
        )?;
        if block.partition != crate::oof::PredictionPartition::Final || block.fold_id.is_some() {
            return contract_error(
                "training replay prediction blocks must use final partition without fold",
            );
        }
    }
    for block in &output.observation_predictions {
        validate_optional_port(
            "training replay observation prediction producer_port",
            &block.producer_port,
        )?;
        if block.partition != crate::oof::PredictionPartition::Final || block.fold_id.is_some() {
            return contract_error(
                "training replay observation prediction blocks must use final partition without fold",
            );
        }
    }
    for block in &output.aggregated_predictions {
        validate_optional_port(
            "training replay aggregated prediction producer_port",
            &block.producer_port,
        )?;
        if block.partition != crate::oof::PredictionPartition::Final || block.fold_id.is_some() {
            return contract_error(
                "training replay aggregated prediction blocks must use final partition without fold",
            );
        }
    }
    Ok(())
}

fn validate_replay_phase(phase: Phase) -> Result<()> {
    if matches!(phase, Phase::Predict | Phase::Explain) {
        Ok(())
    } else {
        contract_error("training replay V1 supports only PREDICT and EXPLAIN")
    }
}

fn validate_sha256(label: &str, value: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        contract_error(format!(
            "{label} fingerprint must be 64 lowercase hexadecimal characters"
        ))
    }
}

fn validate_identifier(label: &str, value: &str) -> Result<()> {
    if !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.' | b':'))
    {
        Ok(())
    } else {
        contract_error(format!("{label} is not a valid DAG-ML identifier"))
    }
}

fn validate_non_empty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        contract_error(format!("{label} must be non-empty"))
    } else {
        Ok(())
    }
}

fn validate_sorted_unique_identifiers(
    label: &str,
    values: &[String],
    require_non_empty: bool,
) -> Result<()> {
    validate_sorted_unique_text(label, values, require_non_empty)?;
    for value in values {
        validate_identifier(label, value)?;
    }
    Ok(())
}

fn validate_sorted_unique_text(
    label: &str,
    values: &[String],
    require_non_empty: bool,
) -> Result<()> {
    if require_non_empty && values.is_empty() {
        return contract_error(format!("{label} must be non-empty"));
    }
    let mut previous: Option<&str> = None;
    for value in values {
        validate_non_empty(label, value)?;
        if previous.is_some_and(|previous| previous >= value.as_str()) {
            return contract_error(format!("{label} must be strictly sorted and unique"));
        }
        previous = Some(value.as_str());
    }
    Ok(())
}

fn validate_sorted_unique_keys<'a>(
    label: &str,
    values: impl Iterator<Item = &'a str>,
    require_non_empty: bool,
) -> Result<()> {
    let values = values.collect::<Vec<_>>();
    if require_non_empty && values.is_empty() {
        return contract_error(format!("{label} must be non-empty"));
    }
    let mut previous: Option<&str> = None;
    for value in values {
        validate_non_empty(label, value)?;
        if previous.is_some_and(|previous| previous >= value) {
            return contract_error(format!("{label} must be strictly sorted and unique"));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_optional_port(label: &str, value: &Option<String>) -> Result<()> {
    match value {
        Some(value) if !value.trim().is_empty() => Ok(()),
        _ => contract_error(format!("{label} must be present and non-empty")),
    }
}

fn validate_diagnostics(diagnostics: &BTreeMap<String, serde_json::Value>) -> Result<()> {
    for (key, value) in diagnostics {
        validate_non_empty("training replay diagnostic key", key)?;
        if !matches!(
            value,
            serde_json::Value::Null
                | serde_json::Value::Bool(_)
                | serde_json::Value::Number(_)
                | serde_json::Value::String(_)
        ) {
            return contract_error("training replay diagnostics must be scalar JSON values");
        }
    }
    Ok(())
}

fn require_count(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        contract_error(format!("{label} does not match replay payload"))
    }
}

fn zero_fingerprint() -> String {
    "0".repeat(64)
}

fn tcv1_fingerprint_without<T: Serialize>(value: &T, field: &str, label: &str) -> Result<String> {
    let json = serde_json::to_string(value)?;
    strict_tcv1_fingerprint_without(&json, field, label)
}

fn strict_tcv1_fingerprint_without(json: &str, field: &str, label: &str) -> Result<String> {
    parse_typed_json(json)
        .and_then(|value| value.fingerprint_without(field))
        .map_err(|error| {
            DagMlError::RuntimeValidation(format!("{label} is outside strict TCV1: {error}"))
        })
}

fn unsupported_version<T>(label: &str, actual: u32, expected: u32) -> Result<T> {
    contract_error(format!(
        "{label} uses unsupported schema_version {actual}, expected {expected}"
    ))
}

fn contract_error<T>(message: impl Into<String>) -> Result<T> {
    Err(DagMlError::CampaignValidation(message.into()))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use super::*;

    fn root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|path| path.parent())
            .expect("core crate is under crates/dag-ml-core")
            .to_path_buf()
    }

    fn fixture(name: &str) -> String {
        fs::read_to_string(
            root()
                .join("examples")
                .join("fixtures")
                .join("training")
                .join("replay")
                .join(name),
        )
        .expect(name)
    }

    fn training_fixture(name: &str) -> String {
        fs::read_to_string(
            root()
                .join("examples")
                .join("fixtures")
                .join("training")
                .join(name),
        )
        .expect(name)
    }

    #[test]
    fn training_replay_contract_fixtures_parse_and_cross_validate() {
        let predict_source =
            TrainingOutcome::from_json(&training_fixture("training_outcome_refit.v1.json"))
                .expect("predict source training outcome");
        let explain_source =
            TrainingOutcome::from_json(&fixture("training_replay_source_outcome_explain.v1.json"))
                .expect("explain source training outcome");
        let predict_request =
            TrainingReplayRequest::from_json(&fixture("training_replay_request_predict.v1.json"))
                .expect("predict request");
        let predict_outcome =
            TrainingReplayOutcome::from_json(&fixture("training_replay_outcome_predict.v1.json"))
                .expect("predict outcome");
        predict_outcome
            .validate_against(&predict_source, &predict_request)
            .expect("predict cross-links");

        let explain_request =
            TrainingReplayRequest::from_json(&fixture("training_replay_request_explain.v1.json"))
                .expect("explain request");
        let explain_outcome =
            TrainingReplayOutcome::from_json(&fixture("training_replay_outcome_explain.v1.json"))
                .expect("explain outcome");
        explain_outcome
            .validate_against(&explain_source, &explain_request)
            .expect("explain cross-links");

        let explain_only = TrainingReplayOutcome::from_json(&fixture(
            "training_replay_outcome_explain_only.v1.json",
        ))
        .expect("explain-only outcome");
        explain_only
            .validate_against(&explain_source, &explain_request)
            .expect("explain-only cross-links");
    }

    #[test]
    fn training_replay_request_rejects_refit_and_unsorted_bindings() {
        let mut request: serde_json::Value =
            serde_json::from_str(&fixture("training_replay_request_predict.v1.json")).unwrap();
        request["phase"] = serde_json::Value::String("REFIT".to_string());
        let err = serde_json::from_value::<TrainingReplayRequest>(request)
            .unwrap()
            .validate()
            .unwrap_err()
            .to_string();
        assert!(err.contains("PREDICT and EXPLAIN"));

        let mut request: TrainingReplayRequest =
            TrainingReplayRequest::from_json(&fixture("training_replay_request_predict.v1.json"))
                .unwrap();
        request.output_binding_ids = vec!["z".to_string(), "a".to_string()];
        request.request_fingerprint = request.compute_fingerprint().unwrap();
        let err = request.validate().unwrap_err().to_string();
        assert!(err.contains("strictly sorted"));
    }

    #[test]
    fn training_replay_outcome_rejects_counter_and_source_transplants() {
        let source =
            TrainingOutcome::from_json(&training_fixture("training_outcome_refit.v1.json"))
                .unwrap();
        let request =
            TrainingReplayRequest::from_json(&fixture("training_replay_request_predict.v1.json"))
                .unwrap();
        let mut outcome =
            TrainingReplayOutcome::from_json(&fixture("training_replay_outcome_predict.v1.json"))
                .unwrap();
        outcome.prediction_block_count += 1;
        outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
        let err = outcome.validate().unwrap_err().to_string();
        assert!(err.contains("prediction_block_count"));

        let mut outcome =
            TrainingReplayOutcome::from_json(&fixture("training_replay_outcome_predict.v1.json"))
                .unwrap();
        outcome.source_training_outcome.outcome_fingerprint = "f".repeat(64);
        outcome.outcome_fingerprint = outcome.compute_fingerprint().unwrap();
        let err = outcome
            .validate_against(&source, &request)
            .unwrap_err()
            .to_string();
        assert!(err.contains("source reference"));
    }
}
