//! Leakage-checked, sample-keyed targets for a second-stage residual learner.
//!
//! The caller supplies the base model's validation predictions from an inner
//! FoldSet over the learner's training universe. The FoldSet must be scoped to
//! the current outer fold during CV; a global OOF set would let outer-validation
//! targets influence residual targets used to fit the learner.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::fold::FoldSet;
use crate::ids::{NodeId, SampleId};
use crate::oof::{
    validate_prediction_blocks_against_folds, validate_producer_oof_coverage, PredictionBlock,
    PredictionPartition,
};

/// An OOF-derived target view in the exact order declared by its FoldSet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualTargetSet {
    pub base_producer: NodeId,
    pub sample_ids: Vec<SampleId>,
    pub values: Vec<Vec<f64>>,
    pub target_names: Vec<String>,
}

/// Derive `observed - base OOF` without joining rows by their array positions.
///
/// Every supplied prediction must be a finite validation block from the same
/// producer and exactly match its declared fold validation membership. The OOF
/// union must cover the whole requested training universe; resampled folds may
/// predict one sample more than once, in which case their predictions are
/// averaged before subtraction. This operation never accepts train predictions.
pub fn derive_residual_targets(
    fold_set: &FoldSet,
    base_producer: &NodeId,
    base_predictions: &[PredictionBlock],
    observed_targets: &BTreeMap<SampleId, Vec<f64>>,
) -> Result<ResidualTargetSet> {
    fold_set.validate()?;
    if base_predictions.is_empty() {
        return Err(DagMlError::OofValidation(
            "residual target derivation requires base validation OOF predictions".to_string(),
        ));
    }

    let mut width = None;
    let mut target_names: Option<Vec<String>> = None;
    for block in base_predictions {
        if &block.producer_node != base_producer {
            return Err(DagMlError::OofValidation(format!(
                "residual target derivation expected base producer `{base_producer}`, got `{}`",
                block.producer_node
            )));
        }
        if block.partition != PredictionPartition::Validation {
            return Err(DagMlError::OofValidation(format!(
                "residual target derivation accepts only validation OOF, got {:?}",
                block.partition
            )));
        }
        let block_width = block.validate_content()?;
        if width
            .replace(block_width)
            .is_some_and(|previous| previous != block_width)
        {
            return Err(DagMlError::OofValidation(
                "residual base predictions have inconsistent target widths".to_string(),
            ));
        }
        if let Some(previous) = &target_names {
            if previous != &block.target_names {
                return Err(DagMlError::OofValidation(
                    "residual base predictions have inconsistent target names".to_string(),
                ));
            }
        } else {
            target_names = Some(block.target_names.clone());
        }
    }
    validate_prediction_blocks_against_folds(fold_set, base_predictions)?;
    let requested = fold_set.sample_ids.iter().cloned().collect::<BTreeSet<_>>();
    let block_refs = base_predictions.iter().collect::<Vec<_>>();
    validate_producer_oof_coverage(
        base_producer,
        &block_refs,
        fold_set.partition_mode,
        Some(&requested),
    )?;

    let width = width.expect("nonempty base predictions have a validated width");
    let mut sums: BTreeMap<SampleId, (Vec<f64>, usize)> = BTreeMap::new();
    for block in base_predictions {
        for (sample, prediction) in block.sample_ids.iter().zip(&block.values) {
            let (sum, count) = sums
                .entry(sample.clone())
                .or_insert_with(|| (vec![0.0; width], 0));
            for (total, value) in sum.iter_mut().zip(prediction) {
                *total += value;
            }
            *count += 1;
        }
    }

    let mut values = Vec::with_capacity(fold_set.sample_ids.len());
    for sample in &fold_set.sample_ids {
        let observed = observed_targets.get(sample).ok_or_else(|| {
            DagMlError::OofValidation(format!(
                "residual target derivation has no observed target for sample `{sample}`"
            ))
        })?;
        if observed.len() != width || observed.iter().any(|value| !value.is_finite()) {
            return Err(DagMlError::OofValidation(format!(
                "residual observed target for sample `{sample}` must have {width} finite value(s)"
            )));
        }
        let (sum, count) = sums
            .get(sample)
            .expect("validated OOF covers all requested samples");
        let residual = observed
            .iter()
            .zip(sum)
            .map(|(target, total)| target - total / *count as f64)
            .collect::<Vec<_>>();
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(DagMlError::OofValidation(format!(
                "residual target for sample `{sample}` is non-finite"
            )));
        }
        values.push(residual);
    }

    Ok(ResidualTargetSet {
        base_producer: base_producer.clone(),
        sample_ids: fold_set.sample_ids.clone(),
        values,
        target_names: target_names.unwrap_or_default(),
    })
}
