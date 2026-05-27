use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::ids::{ControllerId, FoldId, GroupId, NodeId, ObservationId, SampleId, TargetId};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::policy::{AggregationMethod, AggregationPolicy, AggregationWeights, PredictionLevel};
use crate::relation::SampleRelationSet;

pub const AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION: u32 = 1;
pub const AGGREGATION_CONTROLLER_TASK_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/aggregation_controller_task.v1.schema.json";
pub const AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION: u32 = 1;
pub const AGGREGATION_CONTROLLER_RESULT_SCHEMA_ID: &str =
    "https://github.com/GBeurier/dag-ml/schemas/aggregation_controller_result.v1.schema.json";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObservationPredictionBlock {
    #[serde(default)]
    pub prediction_id: Option<String>,
    pub producer_node: NodeId,
    pub partition: PredictionPartition,
    pub fold_id: Option<FoldId>,
    pub observation_ids: Vec<ObservationId>,
    pub values: Vec<Vec<f64>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub weights: Vec<f64>,
    #[serde(default)]
    pub target_names: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "level", content = "id")]
pub enum PredictionUnitId {
    Sample(SampleId),
    Target(TargetId),
    Group(GroupId),
}

impl PredictionUnitId {
    pub fn level(&self) -> PredictionLevel {
        match self {
            Self::Sample(_) => PredictionLevel::Sample,
            Self::Target(_) => PredictionLevel::Target,
            Self::Group(_) => PredictionLevel::Group,
        }
    }
}

impl fmt::Display for PredictionUnitId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sample(id) => write!(f, "sample:{id}"),
            Self::Target(id) => write!(f, "target:{id}"),
            Self::Group(id) => write!(f, "group:{id}"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AggregatedPredictionBlock {
    #[serde(default)]
    pub prediction_id: Option<String>,
    pub producer_node: NodeId,
    pub partition: PredictionPartition,
    pub fold_id: Option<FoldId>,
    pub level: PredictionLevel,
    pub unit_ids: Vec<PredictionUnitId>,
    pub values: Vec<Vec<f64>>,
    #[serde(default)]
    pub target_names: Vec<String>,
}

impl AggregatedPredictionBlock {
    pub fn validate_shape(&self) -> Result<usize> {
        if self.unit_ids.len() != self.values.len() {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` has {} aggregated unit ids but {} prediction rows",
                self.producer_node,
                self.unit_ids.len(),
                self.values.len()
            )));
        }
        if self
            .unit_ids
            .iter()
            .any(|unit_id| unit_id.level() != self.level)
        {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted aggregated units outside level {:?}",
                self.producer_node, self.level
            )));
        }
        let unique = self.unit_ids.iter().collect::<BTreeSet<_>>();
        if unique.len() != self.unit_ids.len() {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted duplicate aggregated unit ids",
                self.producer_node
            )));
        }
        let width = self.values.first().map_or(0, Vec::len);
        if width == 0 {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted empty aggregated prediction rows",
                self.producer_node
            )));
        }
        if self.values.iter().any(|row| row.len() != width) {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted ragged aggregated prediction rows",
                self.producer_node
            )));
        }
        if self.values.iter().flatten().any(|value| !value.is_finite()) {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted non-finite aggregated prediction values",
                self.producer_node
            )));
        }
        if !self.target_names.is_empty() && self.target_names.len() != width {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` has {} aggregated target names for width {}",
                self.producer_node,
                self.target_names.len(),
                width
            )));
        }
        Ok(width)
    }
}

impl ObservationPredictionBlock {
    pub fn validate_shape(&self) -> Result<usize> {
        if self.observation_ids.len() != self.values.len() {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` has {} observation ids but {} prediction rows",
                self.producer_node,
                self.observation_ids.len(),
                self.values.len()
            )));
        }
        let width = self.values.first().map_or(0, Vec::len);
        if width == 0 {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted empty observation prediction rows",
                self.producer_node
            )));
        }
        if self.values.iter().any(|row| row.len() != width) {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted ragged observation prediction rows",
                self.producer_node
            )));
        }
        if self.values.iter().flatten().any(|value| !value.is_finite()) {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted non-finite observation prediction values",
                self.producer_node
            )));
        }
        if !self.weights.is_empty() {
            if self.weights.len() != self.observation_ids.len() {
                return Err(DagMlError::OofValidation(format!(
                    "producer `{}` has {} observation weights but {} observation ids",
                    self.producer_node,
                    self.weights.len(),
                    self.observation_ids.len()
                )));
            }
            if self
                .weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
            {
                return Err(DagMlError::OofValidation(format!(
                    "producer `{}` emitted non-finite or negative observation weights",
                    self.producer_node
                )));
            }
        }
        if !self.target_names.is_empty() && self.target_names.len() != width {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` has {} target names for width {}",
                self.producer_node,
                self.target_names.len(),
                width
            )));
        }
        let unique = self.observation_ids.iter().collect::<BTreeSet<_>>();
        if unique.len() != self.observation_ids.len() {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted duplicate observation predictions",
                self.producer_node
            )));
        }
        Ok(width)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AggregationControllerTask {
    #[serde(default = "default_aggregation_controller_task_schema_version")]
    pub schema_version: u32,
    pub task_id: String,
    pub controller_id: ControllerId,
    pub policy: AggregationPolicy,
    pub input: AggregationControllerInput,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "input_kind", rename_all = "snake_case")]
pub enum AggregationControllerInput {
    ObservationToSample {
        block: ObservationPredictionBlock,
        relations: SampleRelationSet,
        requested_sample_order: Vec<SampleId>,
    },
    SampleToUnit {
        block: PredictionBlock,
        relations: SampleRelationSet,
        requested_unit_order: Vec<PredictionUnitId>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AggregationControllerResult {
    #[serde(default = "default_aggregation_controller_result_schema_version")]
    pub schema_version: u32,
    pub task_id: String,
    pub output: AggregationControllerOutput,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "output_kind", rename_all = "snake_case")]
pub enum AggregationControllerOutput {
    Sample { block: PredictionBlock },
    Unit { block: AggregatedPredictionBlock },
}

impl AggregationControllerTask {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION {
            return Err(DagMlError::OofValidation(format!(
                "aggregation controller task `{}` uses unsupported schema_version {}",
                self.task_id, self.schema_version
            )));
        }
        if self.task_id.trim().is_empty() {
            return Err(DagMlError::OofValidation(
                "aggregation controller task_id is empty".to_string(),
            ));
        }
        self.policy.validate()?;
        if self.policy.method != AggregationMethod::CustomController {
            return Err(DagMlError::OofValidation(format!(
                "aggregation controller task `{}` must use custom_controller method",
                self.task_id
            )));
        }
        let controller = self
            .policy
            .custom_controller
            .as_ref()
            .expect("custom_controller policy validation requires controller spec");
        if controller.controller_id != self.controller_id {
            return Err(DagMlError::OofValidation(format!(
                "aggregation controller task `{}` targets controller `{}` but policy targets `{}`",
                self.task_id, self.controller_id, controller.controller_id
            )));
        }
        match &self.input {
            AggregationControllerInput::ObservationToSample {
                block,
                relations,
                requested_sample_order,
            } => validate_aggregation_controller_observation_input(
                block,
                relations,
                &self.policy,
                requested_sample_order,
            ),
            AggregationControllerInput::SampleToUnit {
                block,
                relations,
                requested_unit_order,
            } => validate_aggregation_controller_sample_input(
                block,
                relations,
                &self.policy,
                requested_unit_order,
            ),
        }
    }
}

impl AggregationControllerResult {
    pub fn validate_for_task(&self, task: &AggregationControllerTask) -> Result<()> {
        task.validate()?;
        if self.schema_version != AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION {
            return Err(DagMlError::OofValidation(format!(
                "aggregation controller result `{}` uses unsupported schema_version {}",
                self.task_id, self.schema_version
            )));
        }
        if self.task_id != task.task_id {
            return Err(DagMlError::OofValidation(format!(
                "aggregation controller result task_id `{}` does not match task `{}`",
                self.task_id, task.task_id
            )));
        }
        match (&task.input, &self.output) {
            (
                AggregationControllerInput::ObservationToSample {
                    block: input_block,
                    requested_sample_order,
                    ..
                },
                AggregationControllerOutput::Sample { block },
            ) => validate_aggregation_controller_sample_output(
                input_block,
                requested_sample_order,
                block,
            ),
            (
                AggregationControllerInput::SampleToUnit {
                    block: input_block,
                    requested_unit_order,
                    ..
                },
                AggregationControllerOutput::Unit { block },
            ) => validate_aggregation_controller_unit_output(
                input_block,
                requested_unit_order,
                task.policy.aggregation_level,
                block,
            ),
            (AggregationControllerInput::ObservationToSample { .. }, _) => {
                Err(DagMlError::OofValidation(format!(
                    "aggregation controller result `{}` must return sample output for observation input",
                    self.task_id
                )))
            }
            (AggregationControllerInput::SampleToUnit { .. }, _) => {
                Err(DagMlError::OofValidation(format!(
                    "aggregation controller result `{}` must return unit output for sample input",
                    self.task_id
                )))
            }
        }
    }
}

fn validate_aggregation_controller_observation_input(
    block: &ObservationPredictionBlock,
    relations: &SampleRelationSet,
    policy: &AggregationPolicy,
    requested_sample_order: &[SampleId],
) -> Result<()> {
    block.validate_shape()?;
    relations.validate()?;
    if policy.aggregation_level != PredictionLevel::Sample {
        return Err(DagMlError::OofValidation(format!(
            "observation aggregation controller task must output sample predictions, got {:?}",
            policy.aggregation_level
        )));
    }
    validate_unique_order(requested_sample_order, "requested_sample_order")?;
    if matches!(
        policy.weights,
        AggregationWeights::ControllerEmitted | AggregationWeights::Quality
    ) && block.weights.is_empty()
    {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller task with {:?} weights requires observation weights",
            policy.weights
        )));
    }
    let requested = requested_sample_order.iter().collect::<BTreeSet<_>>();
    let mut covered = BTreeSet::new();
    for observation_id in &block.observation_ids {
        let sample_id = relations
            .sample_for_observation(observation_id)
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "observation prediction `{observation_id}` has no sample relation"
                ))
            })?;
        if !requested.contains(sample_id) {
            return Err(DagMlError::OofValidation(format!(
                "observation prediction `{observation_id}` maps to unexpected sample `{sample_id}`"
            )));
        }
        covered.insert(sample_id);
    }
    for sample_id in requested_sample_order {
        if !covered.contains(sample_id) {
            return Err(DagMlError::OofValidation(format!(
                "sample `{sample_id}` has no observation predictions for aggregation controller task"
            )));
        }
    }
    Ok(())
}

fn validate_aggregation_controller_sample_input(
    block: &PredictionBlock,
    relations: &SampleRelationSet,
    policy: &AggregationPolicy,
    requested_unit_order: &[PredictionUnitId],
) -> Result<()> {
    validate_sample_prediction_block(block)?;
    relations.validate()?;
    if policy.aggregation_level == PredictionLevel::Observation {
        return Err(DagMlError::OofValidation(
            "sample aggregation controller task cannot output observation-level predictions"
                .to_string(),
        ));
    }
    if matches!(
        policy.weights,
        AggregationWeights::ControllerEmitted | AggregationWeights::Quality
    ) {
        return Err(DagMlError::OofValidation(format!(
            "sample aggregation controller task cannot use {:?} weights without sample weights",
            policy.weights
        )));
    }
    validate_unique_order(requested_unit_order, "requested_unit_order")?;
    if requested_unit_order
        .iter()
        .any(|unit_id| unit_id.level() != policy.aggregation_level)
    {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller requested units do not match level {:?}",
            policy.aggregation_level
        )));
    }
    let requested = requested_unit_order.iter().collect::<BTreeSet<_>>();
    let mut covered = BTreeSet::new();
    for sample_id in &block.sample_ids {
        let unit_id = unit_for_sample(relations, policy.aggregation_level, sample_id)?;
        if !requested.contains(&unit_id) {
            return Err(DagMlError::OofValidation(format!(
                "sample prediction `{sample_id}` maps to unexpected aggregation unit `{unit_id}`"
            )));
        }
        covered.insert(unit_id);
    }
    for unit_id in requested_unit_order {
        if !covered.contains(unit_id) {
            return Err(DagMlError::OofValidation(format!(
                "aggregation unit `{unit_id}` has no sample predictions for aggregation controller task"
            )));
        }
    }
    Ok(())
}

fn validate_aggregation_controller_sample_output(
    input_block: &ObservationPredictionBlock,
    requested_sample_order: &[SampleId],
    block: &PredictionBlock,
) -> Result<()> {
    validate_sample_prediction_block(block)?;
    if block.producer_node != input_block.producer_node
        || block.partition != input_block.partition
        || block.fold_id != input_block.fold_id
    {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller sample output for `{}` does not preserve producer, partition and fold",
            input_block.producer_node
        )));
    }
    if block.target_names != input_block.target_names {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller sample output for `{}` does not preserve target names",
            input_block.producer_node
        )));
    }
    if block.sample_ids != requested_sample_order {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller sample output for `{}` does not match requested sample order",
            input_block.producer_node
        )));
    }
    Ok(())
}

fn validate_aggregation_controller_unit_output(
    input_block: &PredictionBlock,
    requested_unit_order: &[PredictionUnitId],
    expected_level: PredictionLevel,
    block: &AggregatedPredictionBlock,
) -> Result<()> {
    block.validate_shape()?;
    if block.producer_node != input_block.producer_node
        || block.partition != input_block.partition
        || block.fold_id != input_block.fold_id
    {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller unit output for `{}` does not preserve producer, partition and fold",
            input_block.producer_node
        )));
    }
    if block.target_names != input_block.target_names {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller unit output for `{}` does not preserve target names",
            input_block.producer_node
        )));
    }
    if block.level != expected_level {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller unit output for `{}` has level {:?}, expected {:?}",
            input_block.producer_node, block.level, expected_level
        )));
    }
    if block.unit_ids != requested_unit_order {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller unit output for `{}` does not match requested unit order",
            input_block.producer_node
        )));
    }
    Ok(())
}

fn validate_unique_order<T>(values: &[T], label: &str) -> Result<()>
where
    T: Ord,
{
    if values.is_empty() {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller {label} is empty"
        )));
    }
    let unique = values.iter().collect::<BTreeSet<_>>();
    if unique.len() != values.len() {
        return Err(DagMlError::OofValidation(format!(
            "aggregation controller {label} contains duplicates"
        )));
    }
    Ok(())
}

pub fn aggregate_observation_predictions(
    block: &ObservationPredictionBlock,
    relations: &SampleRelationSet,
    policy: &AggregationPolicy,
    requested_sample_order: &[SampleId],
) -> Result<PredictionBlock> {
    let width = block.validate_shape()?;
    relations.validate()?;
    policy.validate()?;
    if requested_sample_order.is_empty() {
        return Err(DagMlError::OofValidation(
            "aggregation requested_sample_order is empty".to_string(),
        ));
    }
    let requested = requested_sample_order.iter().collect::<BTreeSet<_>>();
    if requested.len() != requested_sample_order.len() {
        return Err(DagMlError::OofValidation(
            "aggregation requested_sample_order contains duplicates".to_string(),
        ));
    }
    if policy.aggregation_level != PredictionLevel::Sample {
        return Err(DagMlError::OofValidation(format!(
            "observation aggregation currently supports sample-level output, got {:?}",
            policy.aggregation_level
        )));
    }
    if policy.method == AggregationMethod::WeightedMean
        && policy.weights == AggregationWeights::None
    {
        return Err(DagMlError::OofValidation(
            "weighted_mean aggregation requires an explicit weights policy".to_string(),
        ));
    }
    if policy.method != AggregationMethod::WeightedMean
        && policy.weights != AggregationWeights::None
    {
        return Err(DagMlError::OofValidation(format!(
            "aggregation weights {:?} are only valid with weighted_mean",
            policy.weights
        )));
    }
    if !block.weights.is_empty() && policy.method != AggregationMethod::WeightedMean {
        return Err(DagMlError::OofValidation(format!(
            "producer `{}` supplied observation weights for non-weighted aggregation {:?}",
            block.producer_node, policy.method
        )));
    }

    let store_rows = matches!(
        policy.method,
        AggregationMethod::Median | AggregationMethod::Vote
    );
    let mut accumulators = requested_sample_order
        .iter()
        .cloned()
        .map(|sample_id| (sample_id, SampleAccumulator::new(width, store_rows)))
        .collect::<BTreeMap<_, _>>();

    for (row_idx, (observation_id, row)) in block
        .observation_ids
        .iter()
        .zip(block.values.iter())
        .enumerate()
    {
        let sample_id = relations
            .sample_for_observation(observation_id)
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "observation prediction `{observation_id}` has no sample relation"
                ))
            })?;
        if !requested.contains(sample_id) {
            return Err(DagMlError::OofValidation(format!(
                "observation prediction `{observation_id}` maps to unexpected sample `{sample_id}`"
            )));
        }
        let accumulator = accumulators
            .get_mut(sample_id)
            .expect("requested sample accumulator exists");
        let weight = observation_weight(block, policy, row_idx)?;
        accumulator.push(row, weight);
    }

    let values = requested_sample_order
        .iter()
        .map(|sample_id| {
            let accumulator = accumulators
                .get(sample_id)
                .expect("requested sample accumulator exists");
            if accumulator.count == 0 {
                return Err(DagMlError::OofValidation(format!(
                    "sample `{sample_id}` has no observation predictions to aggregate"
                )));
            }
            match policy.method {
                AggregationMethod::Mean => Ok(accumulator.mean()),
                AggregationMethod::WeightedMean => accumulator.weighted_mean(&sample_id.to_string()),
                AggregationMethod::Median => Ok(accumulator.median()),
                AggregationMethod::Vote => Ok(accumulator.vote()),
                AggregationMethod::None => {
                    if accumulator.count == 1 {
                        Ok(accumulator
                            .first_row
                            .clone()
                            .expect("single prediction accumulator stores first row"))
                    } else {
                        Err(DagMlError::OofValidation(format!(
                            "sample `{sample_id}` has {} observation predictions but aggregation method is none",
                            accumulator.count
                        )))
                    }
                }
                AggregationMethod::CustomController => Err(DagMlError::OofValidation(format!(
                    "aggregation method {:?} is delegated to an aggregation controller",
                    policy.method
                ))),
            }
        })
        .collect::<Result<Vec<Vec<f64>>>>()?;

    Ok(PredictionBlock {
        prediction_id: block
            .prediction_id
            .as_ref()
            .map(|prediction_id| format!("{prediction_id}:sample_agg")),
        producer_node: block.producer_node.clone(),
        partition: block.partition.clone(),
        fold_id: block.fold_id.clone(),
        sample_ids: requested_sample_order.to_vec(),
        values,
        target_names: block.target_names.clone(),
    })
}

pub fn aggregate_sample_predictions_by_unit(
    block: &PredictionBlock,
    relations: &SampleRelationSet,
    policy: &AggregationPolicy,
    requested_unit_order: &[PredictionUnitId],
) -> Result<AggregatedPredictionBlock> {
    let width = validate_sample_prediction_block(block)?;
    relations.validate()?;
    policy.validate()?;
    if requested_unit_order.is_empty() {
        return Err(DagMlError::OofValidation(
            "aggregation requested_unit_order is empty".to_string(),
        ));
    }
    let requested_level = policy.aggregation_level;
    if requested_level == PredictionLevel::Observation {
        return Err(DagMlError::OofValidation(
            "sample prediction aggregation cannot output observation-level predictions".to_string(),
        ));
    }
    if requested_unit_order
        .iter()
        .any(|unit_id| unit_id.level() != requested_level)
    {
        return Err(DagMlError::OofValidation(format!(
            "aggregation requested units do not match level {:?}",
            requested_level
        )));
    }
    let requested = requested_unit_order.iter().collect::<BTreeSet<_>>();
    if requested.len() != requested_unit_order.len() {
        return Err(DagMlError::OofValidation(
            "aggregation requested_unit_order contains duplicates".to_string(),
        ));
    }

    let by_sample = block
        .sample_ids
        .iter()
        .cloned()
        .zip(block.values.iter().cloned())
        .collect::<BTreeMap<_, _>>();
    if requested_level == PredictionLevel::Sample {
        let values = requested_unit_order
            .iter()
            .map(|unit_id| {
                let PredictionUnitId::Sample(sample_id) = unit_id else {
                    unreachable!("requested unit level already validated");
                };
                by_sample.get(sample_id).cloned().ok_or_else(|| {
                    DagMlError::OofValidation(format!(
                        "sample prediction block for `{}` is missing requested sample `{sample_id}`",
                        block.producer_node
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if by_sample.len() != requested_unit_order.len() {
            return Err(DagMlError::OofValidation(format!(
                "sample prediction block for `{}` contains samples outside requested sample order",
                block.producer_node
            )));
        }
        let aggregated = AggregatedPredictionBlock {
            prediction_id: block.prediction_id.clone(),
            producer_node: block.producer_node.clone(),
            partition: block.partition.clone(),
            fold_id: block.fold_id.clone(),
            level: PredictionLevel::Sample,
            unit_ids: requested_unit_order.to_vec(),
            values,
            target_names: block.target_names.clone(),
        };
        aggregated.validate_shape()?;
        return Ok(aggregated);
    }

    if policy.method == AggregationMethod::WeightedMean
        && matches!(
            policy.weights,
            AggregationWeights::ControllerEmitted | AggregationWeights::Quality
        )
    {
        return Err(DagMlError::OofValidation(format!(
            "sample-to-{:?} weighted_mean cannot use {:?} weights without sample-level weights",
            requested_level, policy.weights
        )));
    }

    let store_rows = matches!(
        policy.method,
        AggregationMethod::Median | AggregationMethod::Vote
    );
    let mut accumulators = requested_unit_order
        .iter()
        .cloned()
        .map(|unit_id| (unit_id, SampleAccumulator::new(width, store_rows)))
        .collect::<BTreeMap<_, _>>();

    for (sample_id, row) in block.sample_ids.iter().zip(block.values.iter()) {
        let unit_id = unit_for_sample(relations, requested_level, sample_id)?;
        if !requested.contains(&unit_id) {
            return Err(DagMlError::OofValidation(format!(
                "sample prediction `{sample_id}` maps to unexpected aggregation unit `{unit_id}`"
            )));
        }
        let weight = sample_weight(relations, policy, sample_id)?;
        accumulators
            .get_mut(&unit_id)
            .expect("requested aggregation unit accumulator exists")
            .push(row, weight);
    }

    let values = requested_unit_order
        .iter()
        .map(|unit_id| {
            let accumulator = accumulators
                .get(unit_id)
                .expect("requested aggregation unit accumulator exists");
            if accumulator.count == 0 {
                return Err(DagMlError::OofValidation(format!(
                    "aggregation unit `{unit_id}` has no sample predictions to aggregate"
                )));
            }
            match policy.method {
                AggregationMethod::Mean => Ok(accumulator.mean()),
                AggregationMethod::WeightedMean => accumulator.weighted_mean(&unit_id.to_string()),
                AggregationMethod::Median => Ok(accumulator.median()),
                AggregationMethod::Vote => Ok(accumulator.vote()),
                AggregationMethod::None => {
                    if accumulator.count == 1 {
                        Ok(accumulator
                            .first_row
                            .clone()
                            .expect("single prediction accumulator stores first row"))
                    } else {
                        Err(DagMlError::OofValidation(format!(
                            "aggregation unit `{unit_id}` has {} sample predictions but aggregation method is none",
                            accumulator.count
                        )))
                    }
                }
                AggregationMethod::CustomController => Err(DagMlError::OofValidation(format!(
                    "aggregation method {:?} is delegated to an aggregation controller",
                    policy.method
                ))),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let suffix = match requested_level {
        PredictionLevel::Target => "target_agg",
        PredictionLevel::Group => "group_agg",
        PredictionLevel::Sample => "sample_agg",
        PredictionLevel::Observation => unreachable!("observation output rejected above"),
    };
    let aggregated = AggregatedPredictionBlock {
        prediction_id: block
            .prediction_id
            .as_ref()
            .map(|prediction_id| format!("{prediction_id}:{suffix}")),
        producer_node: block.producer_node.clone(),
        partition: block.partition.clone(),
        fold_id: block.fold_id.clone(),
        level: requested_level,
        unit_ids: requested_unit_order.to_vec(),
        values,
        target_names: block.target_names.clone(),
    };
    aggregated.validate_shape()?;
    Ok(aggregated)
}

fn validate_sample_prediction_block(block: &PredictionBlock) -> Result<usize> {
    let width = block.validate_shape()?;
    if block
        .values
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(DagMlError::OofValidation(format!(
            "producer `{}` emitted non-finite sample prediction values",
            block.producer_node
        )));
    }
    let unique = block.sample_ids.iter().collect::<BTreeSet<_>>();
    if unique.len() != block.sample_ids.len() {
        return Err(DagMlError::OofValidation(format!(
            "producer `{}` emitted duplicate sample predictions",
            block.producer_node
        )));
    }
    Ok(width)
}

fn unit_for_sample(
    relations: &SampleRelationSet,
    level: PredictionLevel,
    sample_id: &SampleId,
) -> Result<PredictionUnitId> {
    match level {
        PredictionLevel::Sample => Ok(PredictionUnitId::Sample(sample_id.clone())),
        PredictionLevel::Target => relations
            .target_for_sample(sample_id)
            .cloned()
            .map(PredictionUnitId::Target)
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "sample `{sample_id}` is missing target id for target aggregation"
                ))
            }),
        PredictionLevel::Group => relations
            .group_for_sample(sample_id)
            .cloned()
            .map(PredictionUnitId::Group)
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "sample `{sample_id}` is missing group id for group aggregation"
                ))
            }),
        PredictionLevel::Observation => Err(DagMlError::OofValidation(
            "sample prediction aggregation cannot output observation-level predictions".to_string(),
        )),
    }
}

fn sample_weight(
    relations: &SampleRelationSet,
    policy: &AggregationPolicy,
    sample_id: &SampleId,
) -> Result<f64> {
    if policy.method != AggregationMethod::WeightedMean {
        return Ok(1.0);
    }
    match policy.weights {
        AggregationWeights::RepetitionCount => {
            let count = relations.observation_count_for_sample(sample_id);
            if count == 0 {
                return Err(DagMlError::OofValidation(format!(
                    "sample `{sample_id}` has no observation relations for repetition_count weights"
                )));
            }
            Ok(count as f64)
        }
        AggregationWeights::ControllerEmitted | AggregationWeights::Quality => {
            Err(DagMlError::OofValidation(format!(
                "sample-level {:?} weights are not present in PredictionBlock",
                policy.weights
            )))
        }
        AggregationWeights::None => Err(DagMlError::OofValidation(
            "weighted_mean aggregation requires an explicit weights policy".to_string(),
        )),
    }
}

#[derive(Clone, Debug)]
struct SampleAccumulator {
    sum: Vec<f64>,
    weighted_sum: Vec<f64>,
    weight_sum: f64,
    rows: Vec<Vec<f64>>,
    first_row: Option<Vec<f64>>,
    store_rows: bool,
    count: usize,
}

impl SampleAccumulator {
    fn new(width: usize, store_rows: bool) -> Self {
        Self {
            sum: vec![0.0; width],
            weighted_sum: vec![0.0; width],
            weight_sum: 0.0,
            rows: Vec::new(),
            first_row: None,
            store_rows,
            count: 0,
        }
    }

    fn push(&mut self, row: &[f64], weight: f64) {
        for (idx, value) in row.iter().enumerate() {
            self.sum[idx] += *value;
            self.weighted_sum[idx] += *value * weight;
        }
        self.weight_sum += weight;
        if self.first_row.is_none() {
            self.first_row = Some(row.to_vec());
        }
        if self.store_rows {
            self.rows.push(row.to_vec());
        }
        self.count += 1;
    }

    fn mean(&self) -> Vec<f64> {
        self.sum
            .iter()
            .map(|value| *value / self.count as f64)
            .collect()
    }

    fn weighted_mean(&self, unit_label: &str) -> Result<Vec<f64>> {
        if self.weight_sum <= 0.0 {
            return Err(DagMlError::OofValidation(format!(
                "aggregation unit `{unit_label}` has zero total prediction weight"
            )));
        }
        Ok(self
            .weighted_sum
            .iter()
            .map(|value| *value / self.weight_sum)
            .collect())
    }

    fn median(&self) -> Vec<f64> {
        let width = self.sum.len();
        (0..width)
            .map(|column_idx| {
                let mut column = self
                    .rows
                    .iter()
                    .map(|row| row[column_idx])
                    .collect::<Vec<_>>();
                column.sort_by(f64::total_cmp);
                let middle = column.len() / 2;
                if column.len() % 2 == 1 {
                    column[middle]
                } else {
                    (column[middle - 1] + column[middle]) / 2.0
                }
            })
            .collect()
    }

    fn vote(&self) -> Vec<f64> {
        let width = self.sum.len();
        (0..width)
            .map(|column_idx| {
                let mut column = self
                    .rows
                    .iter()
                    .map(|row| row[column_idx])
                    .collect::<Vec<_>>();
                column.sort_by(f64::total_cmp);
                mode_sorted(&column)
            })
            .collect()
    }
}

fn observation_weight(
    block: &ObservationPredictionBlock,
    policy: &AggregationPolicy,
    row_idx: usize,
) -> Result<f64> {
    if policy.method != AggregationMethod::WeightedMean {
        return Ok(1.0);
    }
    match policy.weights {
        AggregationWeights::ControllerEmitted | AggregationWeights::Quality => block
            .weights
            .get(row_idx)
            .copied()
            .ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "weighted_mean aggregation with {:?} weights requires one weight per observation",
                    policy.weights
                ))
            }),
        AggregationWeights::RepetitionCount => Ok(1.0),
        AggregationWeights::None => Err(DagMlError::OofValidation(
            "weighted_mean aggregation requires an explicit weights policy".to_string(),
        )),
    }
}

fn mode_sorted(values: &[f64]) -> f64 {
    let mut best_value = values[0];
    let mut best_count = 1usize;
    let mut current_value = values[0];
    let mut current_count = 1usize;
    for value in values.iter().skip(1) {
        if *value == current_value {
            current_count += 1;
            continue;
        }
        if current_count > best_count {
            best_value = current_value;
            best_count = current_count;
        }
        current_value = *value;
        current_count = 1;
    }
    if current_count > best_count {
        current_value
    } else {
        best_value
    }
}

fn default_aggregation_controller_task_schema_version() -> u32 {
    AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION
}

fn default_aggregation_controller_result_schema_version() -> u32 {
    AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::{ControllerId, GroupId, TargetId};
    use crate::relation::SampleRelation;

    fn sid(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }

    fn oid(value: &str) -> ObservationId {
        ObservationId::new(value).unwrap()
    }

    fn relation(observation: &str, sample: &str) -> SampleRelation {
        SampleRelation {
            observation_id: oid(observation),
            sample_id: sid(sample),
            target_id: Some(TargetId::new(format!("target:{sample}")).unwrap()),
            group_id: None,
            origin_sample_id: None,
            source_id: None,
            is_augmented: false,
        }
    }

    fn relation_with_units(
        observation: &str,
        sample: &str,
        target: &str,
        group: &str,
    ) -> SampleRelation {
        SampleRelation {
            observation_id: oid(observation),
            sample_id: sid(sample),
            target_id: Some(TargetId::new(target).unwrap()),
            group_id: Some(GroupId::new(group).unwrap()),
            origin_sample_id: None,
            source_id: None,
            is_augmented: false,
        }
    }

    fn custom_policy(level: PredictionLevel) -> AggregationPolicy {
        AggregationPolicy {
            aggregation_level: level,
            method: AggregationMethod::CustomController,
            custom_controller: Some(crate::policy::AggregationControllerSpec {
                controller_id: ControllerId::new("controller:agg.trimmed").unwrap(),
                params: serde_json::json!({ "trim_fraction": 0.1 }),
            }),
            ..AggregationPolicy::default()
        }
    }

    #[test]
    fn validates_custom_observation_aggregation_controller_result() {
        let task = AggregationControllerTask {
            schema_version: AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION,
            task_id: "agg-task:obs.sample.fold0".to_string(),
            controller_id: ControllerId::new("controller:agg.trimmed").unwrap(),
            policy: custom_policy(PredictionLevel::Sample),
            input: AggregationControllerInput::ObservationToSample {
                block: ObservationPredictionBlock {
                    prediction_id: Some("prediction:model.fold0".to_string()),
                    producer_node: NodeId::new("model:pls").unwrap(),
                    partition: PredictionPartition::Validation,
                    fold_id: Some(FoldId::new("fold:0").unwrap()),
                    observation_ids: vec![oid("obs:1"), oid("obs:2"), oid("obs:3")],
                    values: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![9.0, 10.0]],
                    weights: Vec::new(),
                    target_names: vec!["moisture".to_string(), "protein".to_string()],
                },
                relations: SampleRelationSet {
                    records: vec![
                        relation("obs:1", "sample:1"),
                        relation("obs:2", "sample:1"),
                        relation("obs:3", "sample:2"),
                    ],
                },
                requested_sample_order: vec![sid("sample:1"), sid("sample:2")],
            },
        };
        task.validate().unwrap();

        let result = AggregationControllerResult {
            schema_version: AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION,
            task_id: task.task_id.clone(),
            output: AggregationControllerOutput::Sample {
                block: PredictionBlock {
                    prediction_id: Some("prediction:model.fold0:custom_sample_agg".to_string()),
                    producer_node: NodeId::new("model:pls").unwrap(),
                    partition: PredictionPartition::Validation,
                    fold_id: Some(FoldId::new("fold:0").unwrap()),
                    sample_ids: vec![sid("sample:1"), sid("sample:2")],
                    values: vec![vec![2.0, 3.0], vec![9.0, 10.0]],
                    target_names: vec!["moisture".to_string(), "protein".to_string()],
                },
            },
        };

        result.validate_for_task(&task).unwrap();
    }

    #[test]
    fn custom_aggregation_controller_result_refuses_order_mismatch() {
        let task = AggregationControllerTask {
            schema_version: AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION,
            task_id: "agg-task:obs.sample.fold0".to_string(),
            controller_id: ControllerId::new("controller:agg.trimmed").unwrap(),
            policy: custom_policy(PredictionLevel::Sample),
            input: AggregationControllerInput::ObservationToSample {
                block: ObservationPredictionBlock {
                    prediction_id: None,
                    producer_node: NodeId::new("model:pls").unwrap(),
                    partition: PredictionPartition::Validation,
                    fold_id: None,
                    observation_ids: vec![oid("obs:1"), oid("obs:2")],
                    values: vec![vec![1.0], vec![2.0]],
                    weights: Vec::new(),
                    target_names: vec!["y".to_string()],
                },
                relations: SampleRelationSet {
                    records: vec![relation("obs:1", "sample:1"), relation("obs:2", "sample:2")],
                },
                requested_sample_order: vec![sid("sample:1"), sid("sample:2")],
            },
        };
        let result = AggregationControllerResult {
            schema_version: AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION,
            task_id: task.task_id.clone(),
            output: AggregationControllerOutput::Sample {
                block: PredictionBlock {
                    prediction_id: None,
                    producer_node: NodeId::new("model:pls").unwrap(),
                    partition: PredictionPartition::Validation,
                    fold_id: None,
                    sample_ids: vec![sid("sample:2"), sid("sample:1")],
                    values: vec![vec![2.0], vec![1.0]],
                    target_names: vec!["y".to_string()],
                },
            },
        };

        let error = result.validate_for_task(&task).unwrap_err().to_string();
        assert!(error.contains("requested sample order"));
    }

    #[test]
    fn validates_custom_sample_to_group_aggregation_controller_result() {
        let task = AggregationControllerTask {
            schema_version: AGGREGATION_CONTROLLER_TASK_SCHEMA_VERSION,
            task_id: "agg-task:sample.group.fold0".to_string(),
            controller_id: ControllerId::new("controller:agg.trimmed").unwrap(),
            policy: custom_policy(PredictionLevel::Group),
            input: AggregationControllerInput::SampleToUnit {
                block: PredictionBlock {
                    prediction_id: Some("prediction:model.fold0".to_string()),
                    producer_node: NodeId::new("model:pls").unwrap(),
                    partition: PredictionPartition::Validation,
                    fold_id: Some(FoldId::new("fold:0").unwrap()),
                    sample_ids: vec![sid("sample:1"), sid("sample:2"), sid("sample:3")],
                    values: vec![vec![1.0], vec![3.0], vec![10.0]],
                    target_names: vec!["y".to_string()],
                },
                relations: SampleRelationSet {
                    records: vec![
                        relation_with_units("obs:1", "sample:1", "target:1", "group:left"),
                        relation_with_units("obs:2", "sample:2", "target:2", "group:left"),
                        relation_with_units("obs:3", "sample:3", "target:3", "group:right"),
                    ],
                },
                requested_unit_order: vec![
                    PredictionUnitId::Group(GroupId::new("group:left").unwrap()),
                    PredictionUnitId::Group(GroupId::new("group:right").unwrap()),
                ],
            },
        };
        task.validate().unwrap();

        let result = AggregationControllerResult {
            schema_version: AGGREGATION_CONTROLLER_RESULT_SCHEMA_VERSION,
            task_id: task.task_id.clone(),
            output: AggregationControllerOutput::Unit {
                block: AggregatedPredictionBlock {
                    prediction_id: Some("prediction:model.fold0:custom_group_agg".to_string()),
                    producer_node: NodeId::new("model:pls").unwrap(),
                    partition: PredictionPartition::Validation,
                    fold_id: Some(FoldId::new("fold:0").unwrap()),
                    level: PredictionLevel::Group,
                    unit_ids: vec![
                        PredictionUnitId::Group(GroupId::new("group:left").unwrap()),
                        PredictionUnitId::Group(GroupId::new("group:right").unwrap()),
                    ],
                    values: vec![vec![2.0], vec![10.0]],
                    target_names: vec!["y".to_string()],
                },
            },
        };

        result.validate_for_task(&task).unwrap();
    }

    #[test]
    fn averages_repeated_observation_predictions_by_sample() {
        let block = ObservationPredictionBlock {
            prediction_id: Some("pred:oof".to_string()),
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            observation_ids: vec![oid("obs:1a"), oid("obs:1b"), oid("obs:2a")],
            values: vec![vec![1.0], vec![3.0], vec![10.0]],
            weights: Vec::new(),
            target_names: vec!["y".to_string()],
        };
        let relations = SampleRelationSet {
            records: vec![
                relation("obs:1a", "sample:1"),
                relation("obs:1b", "sample:1"),
                relation("obs:2a", "sample:2"),
            ],
        };

        let aggregated = aggregate_observation_predictions(
            &block,
            &relations,
            &AggregationPolicy::default(),
            &[sid("sample:1"), sid("sample:2")],
        )
        .unwrap();

        assert_eq!(
            aggregated.sample_ids,
            vec![sid("sample:1"), sid("sample:2")]
        );
        assert_eq!(aggregated.values, vec![vec![2.0], vec![10.0]]);
    }

    #[test]
    fn aggregates_repeated_predictions_with_median_vote_and_weights() {
        let relations = SampleRelationSet {
            records: vec![
                relation("obs:1a", "sample:1"),
                relation("obs:1b", "sample:1"),
                relation("obs:1c", "sample:1"),
                relation("obs:2a", "sample:2"),
                relation("obs:2b", "sample:2"),
            ],
        };
        let base_block = ObservationPredictionBlock {
            prediction_id: Some("pred:oof".to_string()),
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            observation_ids: vec![
                oid("obs:1a"),
                oid("obs:1b"),
                oid("obs:1c"),
                oid("obs:2a"),
                oid("obs:2b"),
            ],
            values: vec![
                vec![1.0, 0.0],
                vec![5.0, 1.0],
                vec![9.0, 1.0],
                vec![10.0, 2.0],
                vec![30.0, 3.0],
            ],
            weights: Vec::new(),
            target_names: vec!["regression".to_string(), "class".to_string()],
        };
        let sample_order = [sid("sample:1"), sid("sample:2")];

        let median_policy = AggregationPolicy {
            method: AggregationMethod::Median,
            ..AggregationPolicy::default()
        };
        let median = aggregate_observation_predictions(
            &base_block,
            &relations,
            &median_policy,
            &sample_order,
        )
        .unwrap();
        assert_eq!(median.values, vec![vec![5.0, 1.0], vec![20.0, 2.5]]);

        let vote_policy = AggregationPolicy {
            method: AggregationMethod::Vote,
            ..AggregationPolicy::default()
        };
        let vote =
            aggregate_observation_predictions(&base_block, &relations, &vote_policy, &sample_order)
                .unwrap();
        assert_eq!(vote.values, vec![vec![1.0, 1.0], vec![10.0, 2.0]]);

        let mut weighted_block = base_block;
        weighted_block.weights = vec![1.0, 1.0, 2.0, 1.0, 3.0];
        let weighted_policy = AggregationPolicy {
            method: AggregationMethod::WeightedMean,
            weights: AggregationWeights::ControllerEmitted,
            ..AggregationPolicy::default()
        };
        let weighted = aggregate_observation_predictions(
            &weighted_block,
            &relations,
            &weighted_policy,
            &sample_order,
        )
        .unwrap();
        assert_eq!(weighted.values, vec![vec![6.0, 0.75], vec![25.0, 2.75]]);
    }

    #[test]
    fn refuses_incompatible_observation_weight_contracts() {
        let relations = SampleRelationSet {
            records: vec![
                relation("obs:1a", "sample:1"),
                relation("obs:1b", "sample:1"),
            ],
        };
        let block = ObservationPredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            observation_ids: vec![oid("obs:1a"), oid("obs:1b")],
            values: vec![vec![1.0], vec![2.0]],
            weights: vec![1.0, 2.0],
            target_names: vec!["y".to_string()],
        };

        let mean_error = aggregate_observation_predictions(
            &block,
            &relations,
            &AggregationPolicy::default(),
            &[sid("sample:1")],
        )
        .unwrap_err()
        .to_string();
        assert!(
            mean_error.contains("non-weighted aggregation"),
            "unexpected mean error: {mean_error}"
        );

        let mut missing_weights_block = block;
        missing_weights_block.weights.clear();
        let weighted_error = aggregate_observation_predictions(
            &missing_weights_block,
            &relations,
            &AggregationPolicy {
                method: AggregationMethod::WeightedMean,
                weights: AggregationWeights::ControllerEmitted,
                ..AggregationPolicy::default()
            },
            &[sid("sample:1")],
        )
        .unwrap_err()
        .to_string();
        assert!(
            weighted_error.contains("requires one weight per observation"),
            "unexpected weighted error: {weighted_error}"
        );
    }

    #[test]
    fn aggregates_sample_predictions_to_target_and_group_units() {
        let relations = SampleRelationSet {
            records: vec![
                relation_with_units("obs:s1:a", "sample:1", "target:a", "group:left"),
                relation_with_units("obs:s1:b", "sample:1", "target:a", "group:left"),
                relation_with_units("obs:s2:a", "sample:2", "target:a", "group:left"),
                relation_with_units("obs:s3:a", "sample:3", "target:b", "group:right"),
            ],
        };
        let block = PredictionBlock {
            prediction_id: Some("pred:sample".to_string()),
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new("fold:0").unwrap()),
            sample_ids: vec![sid("sample:1"), sid("sample:2"), sid("sample:3")],
            values: vec![vec![10.0], vec![4.0], vec![30.0]],
            target_names: vec!["y".to_string()],
        };

        let target_policy = AggregationPolicy {
            aggregation_level: PredictionLevel::Target,
            method: AggregationMethod::Mean,
            ..AggregationPolicy::default()
        };
        let by_target = aggregate_sample_predictions_by_unit(
            &block,
            &relations,
            &target_policy,
            &[
                PredictionUnitId::Target(TargetId::new("target:a").unwrap()),
                PredictionUnitId::Target(TargetId::new("target:b").unwrap()),
            ],
        )
        .unwrap();
        assert_eq!(by_target.level, PredictionLevel::Target);
        assert_eq!(by_target.values, vec![vec![7.0], vec![30.0]]);

        let group_policy = AggregationPolicy {
            aggregation_level: PredictionLevel::Group,
            method: AggregationMethod::WeightedMean,
            weights: AggregationWeights::RepetitionCount,
            ..AggregationPolicy::default()
        };
        let by_group = aggregate_sample_predictions_by_unit(
            &block,
            &relations,
            &group_policy,
            &[
                PredictionUnitId::Group(GroupId::new("group:left").unwrap()),
                PredictionUnitId::Group(GroupId::new("group:right").unwrap()),
            ],
        )
        .unwrap();
        assert_eq!(by_group.level, PredictionLevel::Group);
        assert_eq!(by_group.values, vec![vec![8.0], vec![30.0]]);
    }

    #[test]
    fn refuses_target_group_aggregation_without_relation_units() {
        let relations = SampleRelationSet {
            records: vec![SampleRelation {
                observation_id: oid("obs:1"),
                sample_id: sid("sample:1"),
                target_id: None,
                group_id: None,
                origin_sample_id: None,
                source_id: None,
                is_augmented: false,
            }],
        };
        let block = PredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            sample_ids: vec![sid("sample:1")],
            values: vec![vec![1.0]],
            target_names: vec!["y".to_string()],
        };

        let error = aggregate_sample_predictions_by_unit(
            &block,
            &relations,
            &AggregationPolicy {
                aggregation_level: PredictionLevel::Target,
                method: AggregationMethod::Mean,
                ..AggregationPolicy::default()
            },
            &[PredictionUnitId::Target(
                TargetId::new("target:missing").unwrap(),
            )],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("missing target id"),
            "unexpected target aggregation error: {error}"
        );
    }

    #[test]
    fn refuses_missing_observation_relation() {
        let block = ObservationPredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            observation_ids: vec![oid("obs:missing")],
            values: vec![vec![1.0]],
            weights: Vec::new(),
            target_names: vec!["y".to_string()],
        };

        assert!(aggregate_observation_predictions(
            &block,
            &SampleRelationSet::default(),
            &AggregationPolicy::default(),
            &[sid("sample:1")]
        )
        .is_err());
    }
}
