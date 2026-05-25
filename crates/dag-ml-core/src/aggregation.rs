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
    if policy.weights != AggregationWeights::None {
        return Err(DagMlError::OofValidation(format!(
            "aggregation weights {:?} require a controller-emitted weight column",
            policy.weights
        )));
    }

    let mut accumulators = requested_sample_order
        .iter()
        .cloned()
        .map(|sample_id| (sample_id, (vec![0.0; width], 0usize)))
        .collect::<BTreeMap<_, _>>();

    for (observation_id, row) in block.observation_ids.iter().zip(block.values.iter()) {
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
        let (sum, count) = accumulators
            .get_mut(sample_id)
            .expect("requested sample accumulator exists");
        for (idx, value) in row.iter().enumerate() {
            sum[idx] += *value;
        }
        *count += 1;
    }

    let values = requested_sample_order
        .iter()
        .map(|sample_id| {
            let (sum, count) = accumulators
                .get(sample_id)
                .expect("requested sample accumulator exists");
            if *count == 0 {
                return Err(DagMlError::OofValidation(format!(
                    "sample `{sample_id}` has no observation predictions to aggregate"
                )));
            }
            match policy.method {
                AggregationMethod::Mean => Ok(sum.iter().map(|value| *value / *count as f64).collect()),
                AggregationMethod::None => {
                    if *count == 1 {
                        Ok(sum.clone())
                    } else {
                        Err(DagMlError::OofValidation(format!(
                            "sample `{sample_id}` has {count} observation predictions but aggregation method is none"
                        )))
                    }
                }
                AggregationMethod::WeightedMean
                | AggregationMethod::Median
                | AggregationMethod::Vote
                | AggregationMethod::CustomController => Err(DagMlError::OofValidation(format!(
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
    fn refuses_missing_observation_relation() {
        let block = ObservationPredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new("model:pls").unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: None,
            observation_ids: vec![oid("obs:missing")],
            values: vec![vec![1.0]],
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
