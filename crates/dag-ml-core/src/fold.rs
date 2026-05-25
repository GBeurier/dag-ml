use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::ids::{FoldId, GroupId, SampleId};
use crate::rng::SeedContext;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FoldAssignment {
    pub fold_id: FoldId,
    pub train_sample_ids: Vec<SampleId>,
    pub validation_sample_ids: Vec<SampleId>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FoldSet {
    pub id: String,
    pub sample_ids: Vec<SampleId>,
    pub folds: Vec<FoldAssignment>,
    #[serde(default)]
    pub sample_groups: BTreeMap<SampleId, GroupId>,
}

impl FoldSet {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(DagMlError::OofValidation(
                "fold set id is empty".to_string(),
            ));
        }
        if self.sample_ids.is_empty() {
            return Err(DagMlError::OofValidation(
                "fold set contains no samples".to_string(),
            ));
        }
        if self.folds.is_empty() {
            return Err(DagMlError::OofValidation(
                "fold set contains no folds".to_string(),
            ));
        }
        let universe = unique_samples("fold set sample_ids", &self.sample_ids)?;
        if !self.sample_groups.is_empty() {
            for sample_id in self.sample_groups.keys() {
                if !universe.contains(sample_id) {
                    return Err(DagMlError::OofValidation(format!(
                        "sample group map references unknown sample `{sample_id}`"
                    )));
                }
            }
            for sample_id in &self.sample_ids {
                if !self.sample_groups.contains_key(sample_id) {
                    return Err(DagMlError::OofValidation(format!(
                        "sample `{sample_id}` is missing from non-empty group map"
                    )));
                }
            }
        }
        let mut fold_ids = BTreeSet::new();
        let mut validation_counts = self
            .sample_ids
            .iter()
            .cloned()
            .map(|sample_id| (sample_id, 0usize))
            .collect::<BTreeMap<_, _>>();

        for fold in &self.folds {
            if !fold_ids.insert(&fold.fold_id) {
                return Err(DagMlError::OofValidation(format!(
                    "duplicate fold id `{}`",
                    fold.fold_id
                )));
            }
            let train = unique_samples(
                &format!("fold `{}` train_sample_ids", fold.fold_id),
                &fold.train_sample_ids,
            )?;
            let validation = unique_samples(
                &format!("fold `{}` validation_sample_ids", fold.fold_id),
                &fold.validation_sample_ids,
            )?;
            if validation.is_empty() {
                return Err(DagMlError::OofValidation(format!(
                    "fold `{}` has no validation samples",
                    fold.fold_id
                )));
            }
            for sample_id in train.union(&validation) {
                if !universe.contains(sample_id) {
                    return Err(DagMlError::OofValidation(format!(
                        "fold `{}` references unknown sample `{}`",
                        fold.fold_id, sample_id
                    )));
                }
            }
            let overlap = train.intersection(&validation).collect::<Vec<_>>();
            if !overlap.is_empty() {
                return Err(DagMlError::OofValidation(format!(
                    "fold `{}` has train/validation overlap at sample `{}`",
                    fold.fold_id, overlap[0]
                )));
            }
            for sample_id in validation {
                *validation_counts
                    .get_mut(sample_id)
                    .expect("validation sample is in universe") += 1;
            }
            self.validate_group_boundary(fold, &train)?;
        }

        for (sample_id, count) in validation_counts {
            if count != 1 {
                return Err(DagMlError::OofValidation(format!(
                    "sample `{}` appears in validation {} time(s), expected exactly once",
                    sample_id, count
                )));
            }
        }

        Ok(())
    }

    fn validate_group_boundary(
        &self,
        fold: &FoldAssignment,
        train: &BTreeSet<&SampleId>,
    ) -> Result<()> {
        if self.sample_groups.is_empty() {
            return Ok(());
        }
        let train_groups = train
            .iter()
            .filter_map(|sample_id| self.sample_groups.get(*sample_id))
            .collect::<BTreeSet<_>>();
        for sample_id in &fold.validation_sample_ids {
            let Some(group_id) = self.sample_groups.get(sample_id) else {
                continue;
            };
            if train_groups.contains(group_id) {
                return Err(DagMlError::OofValidation(format!(
                    "fold `{}` leaks group `{}` across train/validation",
                    fold.fold_id, group_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KFoldSpec {
    pub n_splits: usize,
    #[serde(default)]
    pub shuffle: bool,
    pub seed: Option<u64>,
}

impl KFoldSpec {
    pub fn split(&self, id: impl Into<String>, samples: &[SampleId]) -> Result<FoldSet> {
        if self.n_splits < 2 {
            return Err(DagMlError::OofValidation(
                "KFold requires at least two splits".to_string(),
            ));
        }
        let unique = unique_samples("KFold samples", samples)?;
        if self.n_splits > unique.len() {
            return Err(DagMlError::OofValidation(format!(
                "KFold n_splits={} exceeds sample count {}",
                self.n_splits,
                unique.len()
            )));
        }
        let ordered = ordered_samples(samples, self.shuffle, self.seed.unwrap_or(0));
        let folds = (0..self.n_splits)
            .map(|fold_idx| {
                let validation = ordered
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, sample_id)| {
                        (idx % self.n_splits == fold_idx).then_some(sample_id.clone())
                    })
                    .collect::<Vec<_>>();
                let validation_set = validation.iter().collect::<BTreeSet<_>>();
                let train = ordered
                    .iter()
                    .filter(|sample_id| !validation_set.contains(sample_id))
                    .cloned()
                    .collect::<Vec<_>>();
                Ok(FoldAssignment {
                    fold_id: FoldId::new(format!("fold{fold_idx}"))?,
                    train_sample_ids: train,
                    validation_sample_ids: validation,
                    metadata: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let fold_set = FoldSet {
            id: id.into(),
            sample_ids: ordered_samples(samples, false, 0),
            folds,
            sample_groups: BTreeMap::new(),
        };
        fold_set.validate()?;
        Ok(fold_set)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GroupKFoldSpec {
    pub n_splits: usize,
}

impl GroupKFoldSpec {
    pub fn split(
        &self,
        id: impl Into<String>,
        sample_groups: &BTreeMap<SampleId, GroupId>,
    ) -> Result<FoldSet> {
        if self.n_splits < 2 {
            return Err(DagMlError::OofValidation(
                "GroupKFold requires at least two splits".to_string(),
            ));
        }
        if sample_groups.is_empty() {
            return Err(DagMlError::OofValidation(
                "GroupKFold requires sample groups".to_string(),
            ));
        }
        let mut groups = BTreeMap::<GroupId, Vec<SampleId>>::new();
        for (sample_id, group_id) in sample_groups {
            groups
                .entry(group_id.clone())
                .or_default()
                .push(sample_id.clone());
        }
        if self.n_splits > groups.len() {
            return Err(DagMlError::OofValidation(format!(
                "GroupKFold n_splits={} exceeds group count {}",
                self.n_splits,
                groups.len()
            )));
        }

        let mut grouped = groups.into_iter().collect::<Vec<_>>();
        grouped.sort_by(|(left_group, left_samples), (right_group, right_samples)| {
            right_samples
                .len()
                .cmp(&left_samples.len())
                .then_with(|| left_group.cmp(right_group))
        });

        let mut fold_validation = vec![Vec::<SampleId>::new(); self.n_splits];
        for (_group_id, mut samples) in grouped {
            samples.sort();
            let fold_idx = fold_validation
                .iter()
                .enumerate()
                .min_by(|(left_idx, left), (right_idx, right)| {
                    left.len()
                        .cmp(&right.len())
                        .then_with(|| left_idx.cmp(right_idx))
                })
                .map(|(idx, _)| idx)
                .expect("at least one fold");
            fold_validation[fold_idx].extend(samples);
        }

        let mut sample_ids = sample_groups.keys().cloned().collect::<Vec<_>>();
        sample_ids.sort();
        let folds = fold_validation
            .into_iter()
            .enumerate()
            .map(|(fold_idx, mut validation)| {
                validation.sort();
                let validation_set = validation.iter().collect::<BTreeSet<_>>();
                let train = sample_ids
                    .iter()
                    .filter(|sample_id| !validation_set.contains(sample_id))
                    .cloned()
                    .collect::<Vec<_>>();
                Ok(FoldAssignment {
                    fold_id: FoldId::new(format!("fold{fold_idx}"))?,
                    train_sample_ids: train,
                    validation_sample_ids: validation,
                    metadata: BTreeMap::new(),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let fold_set = FoldSet {
            id: id.into(),
            sample_ids,
            folds,
            sample_groups: sample_groups.clone(),
        };
        fold_set.validate()?;
        Ok(fold_set)
    }
}

fn unique_samples<'a>(label: &str, samples: &'a [SampleId]) -> Result<BTreeSet<&'a SampleId>> {
    let mut seen = BTreeSet::new();
    for sample_id in samples {
        if !seen.insert(sample_id) {
            return Err(DagMlError::OofValidation(format!(
                "{label} contains duplicate sample `{sample_id}`"
            )));
        }
    }
    Ok(seen)
}

fn ordered_samples(samples: &[SampleId], shuffle: bool, seed: u64) -> Vec<SampleId> {
    let mut ordered = samples.to_vec();
    ordered.sort();
    if shuffle {
        let context = SeedContext::root(seed).child("kfold");
        ordered.sort_by(|left, right| {
            context
                .derive_u64(left.as_str())
                .cmp(&context.derive_u64(right.as_str()))
                .then_with(|| left.cmp(right))
        });
    }
    ordered
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sid(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }

    fn gid(value: &str) -> GroupId {
        GroupId::new(value).unwrap()
    }

    #[test]
    fn kfold_is_deterministic_and_covers_samples_once() {
        let samples = ["s1", "s2", "s3", "s4", "s5", "s6"]
            .into_iter()
            .map(sid)
            .collect::<Vec<_>>();
        let spec = KFoldSpec {
            n_splits: 3,
            shuffle: true,
            seed: Some(42),
        };

        let left = spec.split("kfold", &samples).unwrap();
        let right = spec.split("kfold", &samples).unwrap();

        assert_eq!(left, right);
        left.validate().unwrap();
        for fold in &left.folds {
            assert_eq!(fold.validation_sample_ids.len(), 2);
            assert_eq!(fold.train_sample_ids.len(), 4);
        }
    }

    #[test]
    fn fold_validation_rejects_overlap() {
        let fold_set = FoldSet {
            id: "bad".to_string(),
            sample_ids: vec![sid("s1"), sid("s2")],
            folds: vec![FoldAssignment {
                fold_id: FoldId::new("fold0").unwrap(),
                train_sample_ids: vec![sid("s1")],
                validation_sample_ids: vec![sid("s1")],
                metadata: BTreeMap::new(),
            }],
            sample_groups: BTreeMap::new(),
        };

        assert!(fold_set.validate().is_err());
    }

    #[test]
    fn fold_validation_rejects_partial_group_maps() {
        let fold_set = FoldSet {
            id: "bad-groups".to_string(),
            sample_ids: vec![sid("s1"), sid("s2")],
            folds: vec![FoldAssignment {
                fold_id: FoldId::new("fold0").unwrap(),
                train_sample_ids: vec![sid("s2")],
                validation_sample_ids: vec![sid("s1")],
                metadata: BTreeMap::new(),
            }],
            sample_groups: BTreeMap::from([(sid("s1"), gid("g1"))]),
        };

        assert!(fold_set.validate().is_err());
    }

    #[test]
    fn group_kfold_keeps_groups_out_of_train_validation_overlap() {
        let groups = BTreeMap::from([
            (sid("s1"), gid("g1")),
            (sid("s2"), gid("g1")),
            (sid("s3"), gid("g2")),
            (sid("s4"), gid("g2")),
            (sid("s5"), gid("g3")),
            (sid("s6"), gid("g3")),
        ]);
        let fold_set = GroupKFoldSpec { n_splits: 3 }
            .split("group-kfold", &groups)
            .unwrap();

        fold_set.validate().unwrap();
        for fold in &fold_set.folds {
            let train_groups = fold
                .train_sample_ids
                .iter()
                .map(|sample_id| groups.get(sample_id).unwrap())
                .collect::<BTreeSet<_>>();
            for sample_id in &fold.validation_sample_ids {
                assert!(!train_groups.contains(groups.get(sample_id).unwrap()));
            }
        }
    }
}
