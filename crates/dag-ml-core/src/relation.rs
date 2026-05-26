use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::fold::FoldSet;
use crate::ids::{GroupId, ObservationId, SampleId, TargetId};
use crate::policy::{LeakageUnitPolicy, SplitUnit};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoldPartition {
    Train,
    Validation,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SampleRelation {
    pub observation_id: ObservationId,
    pub sample_id: SampleId,
    #[serde(default)]
    pub target_id: Option<TargetId>,
    #[serde(default)]
    pub group_id: Option<GroupId>,
    #[serde(default)]
    pub origin_sample_id: Option<SampleId>,
    #[serde(default)]
    pub source_id: Option<String>,
    #[serde(default)]
    pub is_augmented: bool,
}

impl SampleRelation {
    fn validate(&self) -> Result<()> {
        if self
            .source_id
            .as_ref()
            .is_some_and(|source| source.trim().is_empty())
        {
            return Err(DagMlError::CampaignValidation(format!(
                "relation `{}` has empty source_id",
                self.observation_id
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SampleRelationSet {
    #[serde(default)]
    pub records: Vec<SampleRelation>,
}

impl SampleRelationSet {
    pub fn validate(&self) -> Result<()> {
        let mut observations = BTreeSet::new();
        let mut sample_targets = BTreeMap::<SampleId, TargetId>::new();
        let mut sample_groups = BTreeMap::<SampleId, GroupId>::new();
        for record in &self.records {
            record.validate()?;
            if !observations.insert(&record.observation_id) {
                return Err(DagMlError::CampaignValidation(format!(
                    "duplicate observation relation `{}`",
                    record.observation_id
                )));
            }
            if let Some(target_id) = &record.target_id {
                if let Some(previous) = sample_targets.get(&record.sample_id) {
                    if previous != target_id {
                        return Err(DagMlError::CampaignValidation(format!(
                            "sample `{}` maps to multiple targets",
                            record.sample_id
                        )));
                    }
                } else {
                    sample_targets.insert(record.sample_id.clone(), target_id.clone());
                }
            }
            if let Some(group_id) = &record.group_id {
                if let Some(previous) = sample_groups.get(&record.sample_id) {
                    if previous != group_id {
                        return Err(DagMlError::CampaignValidation(format!(
                            "sample `{}` maps to multiple groups",
                            record.sample_id
                        )));
                    }
                } else {
                    sample_groups.insert(record.sample_id.clone(), group_id.clone());
                }
            }
        }
        Ok(())
    }

    pub fn validate_against_fold_set(
        &self,
        fold_set: &FoldSet,
        policy: &LeakageUnitPolicy,
    ) -> Result<()> {
        self.validate()?;
        fold_set.validate()?;
        policy.validate()?;

        let universe = fold_set.sample_ids.iter().collect::<BTreeSet<_>>();
        for record in &self.records {
            if !universe.contains(&record.sample_id) {
                return Err(DagMlError::CampaignValidation(format!(
                    "relation `{}` references sample `{}` outside fold set",
                    record.observation_id, record.sample_id
                )));
            }
            if let Some(origin_sample_id) = &record.origin_sample_id {
                if !universe.contains(origin_sample_id) {
                    return Err(DagMlError::CampaignValidation(format!(
                        "relation `{}` references origin sample `{}` outside fold set",
                        record.observation_id, origin_sample_id
                    )));
                }
            }
            if policy.require_group_ids && record.group_id.is_none() {
                return Err(DagMlError::CampaignValidation(format!(
                    "relation `{}` is missing required group id",
                    record.observation_id
                )));
            }
        }

        let sample_to_target = self.sample_targets();
        let sample_to_group = self.sample_groups();
        validate_fold_set_groups_match_relations(fold_set, &sample_to_group)?;

        for fold in &fold_set.folds {
            let partitions = fold
                .train_sample_ids
                .iter()
                .map(|sample_id| (sample_id, FoldPartition::Train))
                .chain(
                    fold.validation_sample_ids
                        .iter()
                        .map(|sample_id| (sample_id, FoldPartition::Validation)),
                )
                .collect::<BTreeMap<_, _>>();

            if policy.forbid_origin_cross_fold {
                for record in &self.records {
                    if let Some(origin_sample_id) = &record.origin_sample_id {
                        let sample_partition =
                            partitions.get(&record.sample_id).ok_or_else(|| {
                                DagMlError::CampaignValidation(format!(
                                    "fold `{}` does not contain sample `{}`",
                                    fold.fold_id, record.sample_id
                                ))
                            })?;
                        let origin_partition =
                            partitions.get(origin_sample_id).ok_or_else(|| {
                                DagMlError::CampaignValidation(format!(
                                    "fold `{}` does not contain origin sample `{}`",
                                    fold.fold_id, origin_sample_id
                                ))
                            })?;
                        if sample_partition != origin_partition {
                            return Err(DagMlError::CampaignValidation(format!(
                                "fold `{}` leaks origin sample `{}` into {:?} sample `{}`",
                                fold.fold_id, origin_sample_id, sample_partition, record.sample_id
                            )));
                        }
                    }
                }
            }

            match policy.split_unit {
                SplitUnit::Observation | SplitUnit::Sample => {}
                SplitUnit::Target => validate_unit_partitions(
                    &fold.fold_id.to_string(),
                    "target",
                    &partitions,
                    &sample_to_target,
                )?,
                SplitUnit::Group => validate_unit_partitions(
                    &fold.fold_id.to_string(),
                    "group",
                    &partitions,
                    &sample_to_group,
                )?,
            }
        }
        Ok(())
    }

    pub fn sample_for_observation(&self, observation_id: &ObservationId) -> Option<&SampleId> {
        self.records
            .iter()
            .find(|record| &record.observation_id == observation_id)
            .map(|record| &record.sample_id)
    }

    pub fn target_for_sample(&self, sample_id: &SampleId) -> Option<&TargetId> {
        self.records
            .iter()
            .find(|record| &record.sample_id == sample_id)
            .and_then(|record| record.target_id.as_ref())
    }

    pub fn group_for_sample(&self, sample_id: &SampleId) -> Option<&GroupId> {
        self.records
            .iter()
            .find(|record| &record.sample_id == sample_id)
            .and_then(|record| record.group_id.as_ref())
    }

    pub fn observation_count_for_sample(&self, sample_id: &SampleId) -> usize {
        self.records
            .iter()
            .filter(|record| &record.sample_id == sample_id)
            .count()
    }

    pub fn sample_targets(&self) -> BTreeMap<SampleId, TargetId> {
        self.records
            .iter()
            .filter_map(|record| {
                record
                    .target_id
                    .as_ref()
                    .map(|target_id| (record.sample_id.clone(), target_id.clone()))
            })
            .collect()
    }

    pub fn sample_groups(&self) -> BTreeMap<SampleId, GroupId> {
        self.records
            .iter()
            .filter_map(|record| {
                record
                    .group_id
                    .as_ref()
                    .map(|group_id| (record.sample_id.clone(), group_id.clone()))
            })
            .collect()
    }
}

fn validate_fold_set_groups_match_relations(
    fold_set: &FoldSet,
    sample_to_group: &BTreeMap<SampleId, GroupId>,
) -> Result<()> {
    for (sample_id, fold_group) in &fold_set.sample_groups {
        if let Some(relation_group) = sample_to_group.get(sample_id) {
            if relation_group != fold_group {
                return Err(DagMlError::CampaignValidation(format!(
                    "sample `{sample_id}` has group `{relation_group}` in relations but `{fold_group}` in fold set"
                )));
            }
        }
    }
    Ok(())
}

fn validate_unit_partitions<Unit: Ord + std::fmt::Display>(
    fold_id: &str,
    label: &str,
    partitions: &BTreeMap<&SampleId, FoldPartition>,
    sample_units: &BTreeMap<SampleId, Unit>,
) -> Result<()> {
    let mut unit_partitions = BTreeMap::<&Unit, FoldPartition>::new();
    for (sample_id, partition) in partitions {
        let Some(unit) = sample_units.get(*sample_id) else {
            return Err(DagMlError::CampaignValidation(format!(
                "fold `{fold_id}` sample `{sample_id}` is missing {label} id"
            )));
        };
        if let Some(previous) = unit_partitions.insert(unit, *partition) {
            if previous != *partition {
                return Err(DagMlError::CampaignValidation(format!(
                    "fold `{fold_id}` leaks {label} `{unit}` across train/validation"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fold::FoldAssignment;

    fn sid(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }

    fn oid(value: &str) -> ObservationId {
        ObservationId::new(value).unwrap()
    }

    fn tid(value: &str) -> TargetId {
        TargetId::new(value).unwrap()
    }

    fn gid(value: &str) -> GroupId {
        GroupId::new(value).unwrap()
    }

    fn fold_set() -> FoldSet {
        FoldSet {
            id: "outer".to_string(),
            sample_ids: vec![sid("s1"), sid("s2"), sid("s3"), sid("s4")],
            folds: vec![
                FoldAssignment {
                    fold_id: crate::ids::FoldId::new("fold:0").unwrap(),
                    train_sample_ids: vec![sid("s3"), sid("s4")],
                    validation_sample_ids: vec![sid("s1"), sid("s2")],
                    metadata: BTreeMap::new(),
                },
                FoldAssignment {
                    fold_id: crate::ids::FoldId::new("fold:1").unwrap(),
                    train_sample_ids: vec![sid("s1"), sid("s2")],
                    validation_sample_ids: vec![sid("s3"), sid("s4")],
                    metadata: BTreeMap::new(),
                },
            ],
            sample_groups: BTreeMap::new(),
        }
    }

    fn relation(observation: &str, sample: &str, target: &str, group: &str) -> SampleRelation {
        SampleRelation {
            observation_id: oid(observation),
            sample_id: sid(sample),
            target_id: Some(tid(target)),
            group_id: Some(gid(group)),
            origin_sample_id: None,
            source_id: None,
            is_augmented: false,
        }
    }

    #[test]
    fn repeated_observations_validate_at_sample_split_unit() {
        let relations = SampleRelationSet {
            records: vec![
                relation("obs:1a", "s1", "t1", "g1"),
                relation("obs:1b", "s1", "t1", "g1"),
                relation("obs:2a", "s2", "t2", "g2"),
                relation("obs:3a", "s3", "t3", "g3"),
                relation("obs:4a", "s4", "t4", "g4"),
            ],
        };

        relations
            .validate_against_fold_set(&fold_set(), &LeakageUnitPolicy::default())
            .unwrap();
    }

    #[test]
    fn target_split_refuses_shared_target_across_fold_boundary() {
        let relations = SampleRelationSet {
            records: vec![
                relation("obs:1", "s1", "same_target", "g1"),
                relation("obs:2", "s2", "t2", "g2"),
                relation("obs:3", "s3", "same_target", "g3"),
                relation("obs:4", "s4", "t4", "g4"),
            ],
        };
        let policy = LeakageUnitPolicy {
            split_unit: SplitUnit::Target,
            ..LeakageUnitPolicy::default()
        };

        assert!(relations
            .validate_against_fold_set(&fold_set(), &policy)
            .is_err());
    }

    #[test]
    fn augmentation_origin_cannot_cross_train_validation_boundary() {
        let mut generated = relation("obs:aug", "s3", "t3", "g3");
        generated.origin_sample_id = Some(sid("s1"));
        generated.is_augmented = true;
        let relations = SampleRelationSet {
            records: vec![
                relation("obs:1", "s1", "t1", "g1"),
                relation("obs:2", "s2", "t2", "g2"),
                generated,
                relation("obs:4", "s4", "t4", "g4"),
            ],
        };

        assert!(relations
            .validate_against_fold_set(&fold_set(), &LeakageUnitPolicy::default())
            .is_err());
    }
}
