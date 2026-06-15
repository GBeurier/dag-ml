use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::campaign::stable_json_fingerprint;
use crate::error::{DagMlError, OofLeakageReport, OofLeakageViolation, Result};
use crate::fold::FoldSet;
use crate::ids::{FoldId, NodeId, SampleId};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionPartition {
    Train,
    Validation,
    Test,
    Final,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionJoinKey {
    SampleId,
}

fn default_prediction_join_key() -> PredictionJoinKey {
    PredictionJoinKey::SampleId
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictionBlock {
    #[serde(default)]
    pub prediction_id: Option<String>,
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OofCampaign {
    pub fold_set: FoldSet,
    pub join_policy: PredictionJoinPolicy,
    pub requested_sample_order: Vec<SampleId>,
    pub prediction_blocks: Vec<PredictionBlock>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PredictionJoinPolicy {
    pub node_id: NodeId,
    #[serde(default = "default_prediction_join_key")]
    pub join_on: PredictionJoinKey,
    #[serde(default)]
    pub allow_train_predictions_as_features: bool,
    #[serde(default)]
    pub include_partitions: Vec<PredictionPartition>,
}

#[derive(Clone, Debug)]
struct ProducerPredictions {
    width: usize,
    target_names: Vec<String>,
    by_sample: BTreeMap<SampleId, Vec<f64>>,
}

pub fn join_oof_features(
    blocks: &[PredictionBlock],
    required_samples: &[SampleId],
) -> Result<OofMatrix> {
    validate_prediction_blocks_are_oof(
        &PredictionJoinPolicy {
            node_id: NodeId::new("prediction_join")?,
            join_on: PredictionJoinKey::SampleId,
            allow_train_predictions_as_features: false,
            include_partitions: vec![PredictionPartition::Validation],
        },
        blocks,
    )?;
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

pub fn join_oof_campaign_features(
    policy: &PredictionJoinPolicy,
    blocks: &[PredictionBlock],
    required_samples: &[SampleId],
) -> Result<OofMatrix> {
    validate_prediction_blocks_are_oof(policy, blocks)?;
    ensure_required_samples(required_samples)?;

    let required = required_samples.iter().collect::<BTreeSet<_>>();
    let included_partitions = effective_partitions(policy);
    let mut producers = BTreeMap::<NodeId, ProducerPredictions>::new();

    for block in blocks {
        if !included_partitions.contains(&block.partition) {
            continue;
        }
        let width = block.validate_shape()?;
        let target_names = normalized_targets(block, width);
        let producer = producers
            .entry(block.producer_node.clone())
            .or_insert_with(|| ProducerPredictions {
                width,
                target_names: target_names.clone(),
                by_sample: BTreeMap::new(),
            });
        if producer.width != width {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` changed prediction width from {} to {}",
                block.producer_node, producer.width, width
            )));
        }
        if producer.target_names != target_names {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` changed target names across folds",
                block.producer_node
            )));
        }

        for (sample_id, values) in block.sample_ids.iter().zip(block.values.iter()) {
            if !required.contains(sample_id) {
                return Err(DagMlError::OofValidation(format!(
                    "producer `{}` emitted unexpected sample `{}`",
                    block.producer_node, sample_id
                )));
            }
            if producer
                .by_sample
                .insert(sample_id.clone(), values.clone())
                .is_some()
            {
                return Err(DagMlError::OofValidation(format!(
                    "producer `{}` emitted duplicate OOF prediction for sample `{}`",
                    block.producer_node, sample_id
                )));
            }
        }
    }

    if producers.is_empty() {
        return Err(DagMlError::OofValidation(
            "no prediction blocks were selected for OOF join".to_string(),
        ));
    }

    for (producer_node, producer) in &producers {
        for sample_id in required_samples {
            if !producer.by_sample.contains_key(sample_id) {
                return Err(DagMlError::OofValidation(format!(
                    "producer `{producer_node}` is missing required sample `{sample_id}`"
                )));
            }
        }
    }

    let producer_predictions = producers.into_iter().collect::<Vec<_>>();
    let columns = producer_predictions
        .iter()
        .flat_map(|(producer_node, producer)| {
            producer
                .target_names
                .iter()
                .map(move |target| format!("{producer_node}__{target}"))
        })
        .collect::<Vec<_>>();
    let values = required_samples
        .iter()
        .map(|sample_id| {
            let mut row = Vec::new();
            for (_producer_node, producer) in &producer_predictions {
                row.extend(
                    producer
                        .by_sample
                        .get(sample_id)
                        .expect("required sample was checked")
                        .iter()
                        .copied(),
                );
            }
            row
        })
        .collect::<Vec<_>>();

    Ok(OofMatrix {
        sample_ids: required_samples.to_vec(),
        columns,
        values,
    })
}

pub fn validate_oof_campaign(campaign: &OofCampaign) -> Result<OofMatrix> {
    campaign.fold_set.validate()?;
    validate_requested_samples_match_fold_set(
        &campaign.requested_sample_order,
        &campaign.fold_set,
    )?;
    validate_prediction_blocks_against_folds(&campaign.fold_set, &campaign.prediction_blocks)?;
    join_oof_campaign_features(
        &campaign.join_policy,
        &campaign.prediction_blocks,
        &campaign.requested_sample_order,
    )
}

pub fn oof_campaign_fingerprint(campaign: &OofCampaign) -> Result<String> {
    campaign.fold_set.validate()?;
    validate_requested_samples_match_fold_set(
        &campaign.requested_sample_order,
        &campaign.fold_set,
    )?;
    validate_prediction_blocks_against_folds(&campaign.fold_set, &campaign.prediction_blocks)?;
    stable_json_fingerprint(campaign)
}

pub fn validate_prediction_blocks_against_folds(
    fold_set: &FoldSet,
    blocks: &[PredictionBlock],
) -> Result<()> {
    fold_set.validate()?;
    let folds = fold_set
        .folds
        .iter()
        .map(|fold| (&fold.fold_id, fold))
        .collect::<BTreeMap<_, _>>();
    for block in blocks {
        block.validate_shape()?;
        let Some(fold_id) = &block.fold_id else {
            if matches!(
                block.partition,
                PredictionPartition::Train | PredictionPartition::Validation
            ) {
                return Err(DagMlError::OofValidation(format!(
                    "producer `{}` emitted {:?} predictions without fold_id",
                    block.producer_node, block.partition
                )));
            }
            continue;
        };
        let fold = folds.get(fold_id).ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "producer `{}` references unknown fold `{fold_id}`",
                block.producer_node
            ))
        })?;
        match block.partition {
            PredictionPartition::Train => {
                assert_exact_partition_samples(block, &fold.train_sample_ids, "train")?
            }
            PredictionPartition::Validation => {
                assert_exact_partition_samples(block, &fold.validation_sample_ids, "validation")?
            }
            PredictionPartition::Test | PredictionPartition::Final => {}
        }
    }
    Ok(())
}

pub fn validate_prediction_blocks_are_oof(
    policy: &PredictionJoinPolicy,
    blocks: &[PredictionBlock],
) -> Result<()> {
    if policy.allow_train_predictions_as_features {
        return Ok(());
    }
    let violators = blocks
        .iter()
        .filter(|block| block.partition != PredictionPartition::Validation)
        .map(|block| OofLeakageViolation {
            producer_node: block.producer_node.to_string(),
            partition: format!("{:?}", block.partition).to_lowercase(),
            fold_id: block.fold_id.as_ref().map(ToString::to_string),
        })
        .collect::<Vec<_>>();
    if violators.is_empty() {
        Ok(())
    } else {
        crate::observability::emit_oof_refusal(policy.node_id.as_str(), violators.len());
        Err(DagMlError::OofLeakage(Box::new(OofLeakageReport {
            node_id: policy.node_id.to_string(),
            violators,
            allow_train_predictions_as_features: policy.allow_train_predictions_as_features,
            remediation: "Use only OOF validation predictions as training features, or explicitly set allow_train_predictions_as_features=true for an unsafe run.".to_string(),
        })))
    }
}

fn validate_requested_samples_match_fold_set(
    requested_sample_order: &[SampleId],
    fold_set: &FoldSet,
) -> Result<()> {
    ensure_required_samples(requested_sample_order)?;
    let requested = requested_sample_order.iter().collect::<BTreeSet<_>>();
    let expected = fold_set.sample_ids.iter().collect::<BTreeSet<_>>();
    if requested != expected {
        return Err(DagMlError::OofValidation(
            "requested sample order does not match fold-set sample universe".to_string(),
        ));
    }
    Ok(())
}

fn assert_exact_partition_samples(
    block: &PredictionBlock,
    expected_samples: &[SampleId],
    partition_name: &str,
) -> Result<()> {
    let actual = unique_block_samples(block)?;
    let expected = expected_samples.iter().collect::<BTreeSet<_>>();
    if actual != expected {
        return Err(DagMlError::OofValidation(format!(
            "producer `{}` fold `{}` {} predictions do not match fold {} samples",
            block.producer_node,
            block.fold_id.as_ref().expect("fold id exists"),
            partition_name,
            partition_name
        )));
    }
    Ok(())
}

fn unique_block_samples(block: &PredictionBlock) -> Result<BTreeSet<&SampleId>> {
    let mut seen = BTreeSet::new();
    for sample_id in &block.sample_ids {
        if !seen.insert(sample_id) {
            return Err(DagMlError::OofValidation(format!(
                "producer `{}` emitted duplicate prediction for sample `{sample_id}`",
                block.producer_node
            )));
        }
    }
    Ok(seen)
}

fn ensure_required_samples(required_samples: &[SampleId]) -> Result<()> {
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
    Ok(())
}

fn effective_partitions(policy: &PredictionJoinPolicy) -> BTreeSet<PredictionPartition> {
    if policy.include_partitions.is_empty() {
        BTreeSet::from([PredictionPartition::Validation])
    } else {
        policy.include_partitions.iter().cloned().collect()
    }
}

fn normalized_targets(block: &PredictionBlock, width: usize) -> Vec<String> {
    if block.target_names.is_empty() {
        (0..width)
            .map(|column_idx| format!("p{column_idx}"))
            .collect()
    } else {
        block.target_names.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn sid(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }

    fn producer() -> NodeId {
        NodeId::new("model:base").unwrap()
    }

    fn block(partition: PredictionPartition) -> PredictionBlock {
        PredictionBlock {
            prediction_id: None,
            producer_node: producer(),
            partition,
            fold_id: Some(FoldId::new("fold0").unwrap()),
            sample_ids: vec![sid("s2"), sid("s1")],
            values: vec![vec![20.0], vec![10.0]],
            target_names: vec!["y".to_string()],
        }
    }

    fn campaign_block(producer_node: &str, fold_id: &str, samples: &[&str]) -> PredictionBlock {
        PredictionBlock {
            prediction_id: None,
            producer_node: NodeId::new(producer_node).unwrap(),
            partition: PredictionPartition::Validation,
            fold_id: Some(FoldId::new(fold_id).unwrap()),
            sample_ids: samples.iter().copied().map(sid).collect(),
            values: samples
                .iter()
                .map(|sample_id| {
                    let suffix = sample_id.trim_start_matches('s').parse::<f64>().unwrap();
                    vec![suffix]
                })
                .collect(),
            target_names: vec!["y".to_string()],
        }
    }

    fn load_fixture(source: &str) -> OofCampaign {
        serde_json::from_str(source).unwrap()
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

        match err {
            DagMlError::OofLeakage(report) => {
                assert_eq!(report.violators[0].producer_node, "model:base");
                assert_eq!(report.violators[0].partition, "train");
            }
            other => panic!("expected OOF leakage error, got {other:?}"),
        }
    }

    #[test]
    fn rejects_duplicate_samples() {
        let mut duplicate = block(PredictionPartition::Validation);
        duplicate.sample_ids = vec![sid("s1"), sid("s1")];

        assert!(join_oof_features(&[duplicate], &[sid("s1")]).is_err());
    }

    #[test]
    fn joins_fold_blocks_by_producer_for_campaigns() {
        let mut b1_fold0 = campaign_block("branch:b1.model:rf", "fold0", &["s4", "s1"]);
        b1_fold0.values = vec![vec![40.0], vec![10.0]];
        let mut b1_fold1 = campaign_block("branch:b1.model:rf", "fold1", &["s2", "s3"]);
        b1_fold1.values = vec![vec![20.0], vec![30.0]];
        let mut b0_fold0 = campaign_block("branch:b0.model:pls", "fold0", &["s4", "s1"]);
        b0_fold0.values = vec![vec![4.0], vec![1.0]];
        let mut b0_fold1 = campaign_block("branch:b0.model:pls", "fold1", &["s2", "s3"]);
        b0_fold1.values = vec![vec![2.0], vec![3.0]];

        let joined = join_oof_campaign_features(
            &PredictionJoinPolicy {
                node_id: NodeId::new("merge:pred").unwrap(),
                join_on: PredictionJoinKey::SampleId,
                allow_train_predictions_as_features: false,
                include_partitions: vec![PredictionPartition::Validation],
            },
            &[b1_fold0, b1_fold1, b0_fold0, b0_fold1],
            &[sid("s1"), sid("s2"), sid("s3"), sid("s4")],
        )
        .unwrap();

        assert_eq!(
            joined.columns,
            vec!["branch:b0.model:pls__y", "branch:b1.model:rf__y"]
        );
        assert_eq!(
            joined.values,
            vec![
                vec![1.0, 10.0],
                vec![2.0, 20.0],
                vec![3.0, 30.0],
                vec![4.0, 40.0]
            ]
        );
    }

    #[test]
    fn uc6_fixture_joins_successfully() {
        let fixture = load_fixture(include_str!(
            "../../../examples/fixtures/oof_campaign/uc6_oof_success_predictions.json"
        ));

        let joined = validate_oof_campaign(&fixture).unwrap();
        assert_eq!(
            oof_campaign_fingerprint(&fixture).unwrap(),
            oof_campaign_fingerprint(&fixture).unwrap()
        );

        assert_eq!(joined.columns.len(), 3);
        assert_eq!(joined.values[0], vec![1.0, 10.0, 100.0]);
        assert_eq!(joined.values[5], vec![6.0, 60.0, 600.0]);
    }

    #[test]
    fn uc11_fixture_refuses_train_predictions() {
        let fixture = load_fixture(include_str!(
            "../../../examples/fixtures/oof_campaign/uc11_train_prediction_refusal.json"
        ));

        let err = validate_oof_campaign(&fixture).unwrap_err();

        match err {
            DagMlError::OofLeakage(report) => {
                assert_eq!(report.node_id, "merge:pred");
                assert!(!report.allow_train_predictions_as_features);
                assert_eq!(report.violators.len(), 1);
                assert_eq!(report.violators[0].partition, "train");
            }
            other => panic!("expected OOF leakage error, got {other:?}"),
        }
    }

    #[test]
    fn fold_validation_rejects_wrong_validation_partition_samples() {
        let mut fixture = load_fixture(include_str!(
            "../../../examples/fixtures/oof_campaign/uc6_oof_success_predictions.json"
        ));
        fixture.prediction_blocks[0].sample_ids = vec![sid("S001"), sid("S002")];

        let err = validate_oof_campaign(&fixture).unwrap_err();

        assert!(err
            .to_string()
            .contains("do not match fold validation samples"));
    }

    #[test]
    #[ignore = "perf sanity probe; run with --release --ignored --nocapture"]
    fn oof_join_large_campaign_under_1500ms() {
        let sample_count = 12_000usize;
        let producer_count = 4usize;
        let fold_count = 6usize;
        let required_samples = (0..sample_count)
            .map(|sample_idx| sid(&format!("s{sample_idx:05}")))
            .collect::<Vec<_>>();
        let mut blocks = Vec::new();

        for producer_idx in 0..producer_count {
            for fold_idx in 0..fold_count {
                let sample_ids = (fold_idx..sample_count)
                    .step_by(fold_count)
                    .map(|sample_idx| sid(&format!("s{sample_idx:05}")))
                    .collect::<Vec<_>>();
                let values = (fold_idx..sample_count)
                    .step_by(fold_count)
                    .map(|sample_idx| vec![producer_idx as f64, sample_idx as f64])
                    .collect::<Vec<_>>();
                blocks.push(PredictionBlock {
                    prediction_id: None,
                    producer_node: NodeId::new(format!("model:p{producer_idx}")).unwrap(),
                    partition: PredictionPartition::Validation,
                    fold_id: Some(FoldId::new(format!("fold:{fold_idx}")).unwrap()),
                    sample_ids,
                    values,
                    target_names: vec!["score".to_string(), "rank".to_string()],
                });
            }
        }

        let started = Instant::now();
        let joined = join_oof_campaign_features(
            &PredictionJoinPolicy {
                node_id: NodeId::new("merge:perf").unwrap(),
                join_on: PredictionJoinKey::SampleId,
                allow_train_predictions_as_features: false,
                include_partitions: vec![PredictionPartition::Validation],
            },
            &blocks,
            &required_samples,
        )
        .unwrap();
        let elapsed = started.elapsed();

        assert_eq!(joined.sample_ids.len(), sample_count);
        assert_eq!(joined.columns.len(), producer_count * 2);
        assert!(
            elapsed <= Duration::from_millis(1_500),
            "large OOF join took {elapsed:?}"
        );
    }
}
