use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::ids::{FoldId, NodeId, ObservationId, SampleId};
use crate::oof::{PredictionBlock, PredictionPartition};
use crate::policy::{AggregationMethod, AggregationPolicy, AggregationWeights, PredictionLevel};
use crate::relation::SampleRelationSet;

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
                AggregationMethod::WeightedMean => accumulator.weighted_mean(sample_id),
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

    fn weighted_mean(&self, sample_id: &SampleId) -> Result<Vec<f64>> {
        if self.weight_sum <= 0.0 {
            return Err(DagMlError::OofValidation(format!(
                "sample `{sample_id}` has zero total observation weight"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ids::TargetId;
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
