use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::ids::{FoldId, NodeId, SampleId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionPartition {
    Train,
    Validation,
    Test,
    Final,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionBlock {
    pub producer_node: NodeId,
    pub partition: PredictionPartition,
    pub fold_id: Option<FoldId>,
    pub sample_ids: Vec<SampleId>,
    pub values: Vec<Vec<f64>>,
    #[serde(default)]
    pub target_names: Vec<String>,
}

impl PredictionBlock {
    pub fn validate_shape(&self) -> Result<usize> {
        if self.sample_ids.len() != self.values.len() {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` has {} sample ids but {} prediction rows",
                self.producer_node,
                self.sample_ids.len(),
                self.values.len()
            )));
        }
        let width = self.values.first().map_or(0, Vec::len);
        if width == 0 {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted empty prediction rows",
                self.producer_node
            )));
        }
        if self.values.iter().any(|row| row.len() != width) {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted ragged prediction rows",
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
        Ok(width)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OofMatrix {
    pub sample_ids: Vec<SampleId>,
    pub columns: Vec<String>,
    pub values: Vec<Vec<f64>>,
}

pub fn join_oof_features(
    blocks: &[PredictionBlock],
    required_samples: &[SampleId],
) -> Result<OofMatrix> {
    if required_samples.is_empty() {
        return Err(DagMlError::OofValidation(
            "required sample set is empty".to_string(),
        ));
    }

    let required = required_samples.iter().collect::<BTreeSet<_>>();
    if required.len() != required_samples.len() {
        return Err(DagMlError::OofValidation(
            "required sample set contains duplicates".to_string(),
        ));
    }

    let mut rows = required_samples
        .iter()
        .cloned()
        .map(|sample_id| (sample_id, Vec::<f64>::new()))
        .collect::<BTreeMap<_, _>>();
    let mut columns = Vec::new();

    for block in blocks {
        if block.partition != PredictionPartition::Validation {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted {:?}; OOF training features require validation predictions",
                block.producer_node, block.partition
            )));
        }

        let width = block.validate_shape()?;
        let mut seen = BTreeSet::new();
        let mut by_sample = BTreeMap::new();
        for (sample_id, values) in block.sample_ids.iter().zip(block.values.iter()) {
            if !seen.insert(sample_id) {
                return Err(DagMlError::OofValidation(format!(
                    "producer `{}` emitted duplicate prediction for sample `{}`",
                    block.producer_node, sample_id
                )));
            }
            by_sample.insert(sample_id, values);
        }

        for sample_id in required_samples {
            let values = by_sample.get(sample_id).ok_or_else(|| {
                DagMlError::OofValidation(format!(
                    "producer `{}` is missing required sample `{}`",
                    block.producer_node, sample_id
                ))
            })?;
            rows.get_mut(sample_id)
                .expect("required sample row exists")
                .extend(values.iter().copied());
        }

        for column_idx in 0..width {
            let target = block
                .target_names
                .get(column_idx)
                .cloned()
                .unwrap_or_else(|| format!("p{column_idx}"));
            columns.push(format!("{}__{target}", block.producer_node));
        }
    }

    Ok(OofMatrix {
        sample_ids: required_samples.to_vec(),
        columns,
        values: required_samples
            .iter()
            .map(|sample_id| rows.remove(sample_id).expect("row exists"))
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }

    fn producer() -> NodeId {
        NodeId::new("model:base").unwrap()
    }

    fn block(partition: PredictionPartition) -> PredictionBlock {
        PredictionBlock {
            producer_node: producer(),
            partition,
            fold_id: Some(FoldId::new("fold0").unwrap()),
            sample_ids: vec![sid("s2"), sid("s1")],
            values: vec![vec![20.0], vec![10.0]],
            target_names: vec!["y".to_string()],
        }
    }

    #[test]
    fn aligns_oof_by_sample_id_not_position() {
        let joined = join_oof_features(
            &[block(PredictionPartition::Validation)],
            &[sid("s1"), sid("s2")],
        )
        .unwrap();

        assert_eq!(joined.values, vec![vec![10.0], vec![20.0]]);
        assert_eq!(joined.columns, vec!["model:base__y"]);
    }

    #[test]
    fn rejects_train_predictions_as_training_features() {
        let err = join_oof_features(
            &[block(PredictionPartition::Train)],
            &[sid("s1"), sid("s2")],
        )
        .unwrap_err();

        assert!(err.to_string().contains("require validation predictions"));
    }

    #[test]
    fn rejects_duplicate_samples() {
        let mut duplicate = block(PredictionPartition::Validation);
        duplicate.sample_ids = vec![sid("s1"), sid("s1")];

        assert!(join_oof_features(&[duplicate], &[sid("s1")]).is_err());
    }
}
