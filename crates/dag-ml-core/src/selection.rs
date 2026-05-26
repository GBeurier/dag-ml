use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DagMlError, Result};
use crate::policy::PredictionLevel;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricObjective {
    Minimize,
    Maximize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SelectionMetric {
    pub name: String,
    pub objective: MetricObjective,
}

impl SelectionMetric {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "selection metric name is empty".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateScore {
    pub candidate_id: String,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl CandidateScore {
    pub fn validate(&self) -> Result<()> {
        if self.candidate_id.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "candidate id is empty".to_string(),
            ));
        }
        for (name, value) in &self.metrics {
            if name.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "candidate `{}` has an empty metric name",
                    self.candidate_id
                )));
            }
            if value.is_nan() {
                return Err(DagMlError::CampaignValidation(format!(
                    "candidate `{}` metric `{name}` is NaN",
                    self.candidate_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SelectionPolicy {
    pub id: String,
    pub metric: SelectionMetric,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_metric_level: Option<PredictionLevel>,
    #[serde(default = "default_true")]
    pub require_finite: bool,
}

impl SelectionPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "selection policy id is empty".to_string(),
            ));
        }
        self.metric.validate()
    }
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedCandidate {
    pub candidate_id: String,
    pub score: f64,
    pub rank: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectionDecision {
    pub policy_id: String,
    pub selected_candidate_id: String,
    pub metric_name: String,
    pub objective: MetricObjective,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_level: Option<PredictionLevel>,
    pub selected_score: f64,
    #[serde(default)]
    pub ranked_candidates: Vec<RankedCandidate>,
}

impl SelectionDecision {
    pub fn validate(&self) -> Result<()> {
        if self.policy_id.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "selection decision policy_id is empty".to_string(),
            ));
        }
        if self.selected_candidate_id.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "selection decision selected_candidate_id is empty".to_string(),
            ));
        }
        if self.metric_name.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "selection decision metric_name is empty".to_string(),
            ));
        }
        if !self.selected_score.is_finite() {
            return Err(DagMlError::CampaignValidation(format!(
                "selection `{}` selected score is not finite",
                self.policy_id
            )));
        }
        if self.ranked_candidates.is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "selection `{}` has no ranked candidates",
                self.policy_id
            )));
        }
        if self.ranked_candidates[0].candidate_id != self.selected_candidate_id {
            return Err(DagMlError::CampaignValidation(format!(
                "selection `{}` first ranked candidate does not match selected candidate",
                self.policy_id
            )));
        }
        let mut seen = BTreeSet::new();
        for (idx, candidate) in self.ranked_candidates.iter().enumerate() {
            if candidate.rank != idx + 1 {
                return Err(DagMlError::CampaignValidation(format!(
                    "selection `{}` candidate `{}` has rank {}, expected {}",
                    self.policy_id,
                    candidate.candidate_id,
                    candidate.rank,
                    idx + 1
                )));
            }
            if !seen.insert(candidate.candidate_id.as_str()) {
                return Err(DagMlError::CampaignValidation(format!(
                    "selection `{}` contains duplicate candidate `{}`",
                    self.policy_id, candidate.candidate_id
                )));
            }
        }
        Ok(())
    }
}

pub fn select_candidate(
    policy: &SelectionPolicy,
    candidates: &[CandidateScore],
) -> Result<SelectionDecision> {
    policy.validate()?;
    if candidates.is_empty() {
        return Err(DagMlError::CampaignValidation(format!(
            "selection policy `{}` has no candidates",
            policy.id
        )));
    }

    let mut scored = Vec::with_capacity(candidates.len());
    let mut seen = BTreeSet::new();
    for candidate in candidates {
        candidate.validate()?;
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(DagMlError::CampaignValidation(format!(
                "selection policy `{}` has duplicate candidate `{}`",
                policy.id, candidate.candidate_id
            )));
        }
        validate_candidate_metric_level(policy, candidate)?;
        let score = candidate
            .metrics
            .get(&policy.metric.name)
            .copied()
            .ok_or_else(|| {
                DagMlError::CampaignValidation(format!(
                    "candidate `{}` is missing selection metric `{}`",
                    candidate.candidate_id, policy.metric.name
                ))
            })?;
        if policy.require_finite && !score.is_finite() {
            return Err(DagMlError::CampaignValidation(format!(
                "candidate `{}` metric `{}` is not finite",
                candidate.candidate_id, policy.metric.name
            )));
        }
        scored.push((candidate.candidate_id.clone(), score));
    }

    scored.sort_by(|left, right| compare_scores(policy.metric.objective, left, right));
    let ranked_candidates = scored
        .iter()
        .enumerate()
        .map(|(idx, (candidate_id, score))| RankedCandidate {
            candidate_id: candidate_id.clone(),
            score: *score,
            rank: idx + 1,
        })
        .collect::<Vec<_>>();
    let selected = ranked_candidates
        .first()
        .expect("candidates were checked as non-empty");
    let decision = SelectionDecision {
        policy_id: policy.id.clone(),
        selected_candidate_id: selected.candidate_id.clone(),
        metric_name: policy.metric.name.clone(),
        objective: policy.metric.objective,
        metric_level: policy.required_metric_level,
        selected_score: selected.score,
        ranked_candidates,
    };
    decision.validate()?;
    Ok(decision)
}

pub fn select_candidate_groups(
    policy: &SelectionPolicy,
    candidates: &[CandidateScore],
    groups: &BTreeMap<String, Vec<String>>,
) -> Result<BTreeMap<String, SelectionDecision>> {
    policy.validate()?;
    let mut by_id = BTreeMap::new();
    for candidate in candidates {
        candidate.validate()?;
        if by_id
            .insert(candidate.candidate_id.as_str(), candidate)
            .is_some()
        {
            return Err(DagMlError::CampaignValidation(format!(
                "selection policy `{}` has duplicate candidate `{}`",
                policy.id, candidate.candidate_id
            )));
        }
    }
    let mut decisions = BTreeMap::new();
    for (group_id, candidate_ids) in groups {
        if group_id.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "selection group id is empty".to_string(),
            ));
        }
        if candidate_ids.is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "selection group `{group_id}` has no candidates"
            )));
        }
        let group_candidates = candidate_ids
            .iter()
            .map(|candidate_id| {
                by_id
                    .get(candidate_id.as_str())
                    .cloned()
                    .cloned()
                    .ok_or_else(|| {
                        DagMlError::CampaignValidation(format!(
                        "selection group `{group_id}` references unknown candidate `{candidate_id}`"
                    ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        decisions.insert(
            group_id.clone(),
            select_candidate(policy, &group_candidates)?,
        );
    }
    Ok(decisions)
}

fn compare_scores(
    objective: MetricObjective,
    left: &(String, f64),
    right: &(String, f64),
) -> Ordering {
    let score_order = match objective {
        MetricObjective::Minimize => left.1.total_cmp(&right.1),
        MetricObjective::Maximize => right.1.total_cmp(&left.1),
    };
    score_order.then_with(|| left.0.cmp(&right.0))
}

fn validate_candidate_metric_level(
    policy: &SelectionPolicy,
    candidate: &CandidateScore,
) -> Result<()> {
    let Some(required_level) = policy.required_metric_level else {
        return Ok(());
    };
    let Some(raw_level) = candidate.metadata.get("metric_level") else {
        return Err(DagMlError::CampaignValidation(format!(
            "candidate `{}` is missing required metric_level `{}`",
            candidate.candidate_id,
            prediction_level_name(required_level)
        )));
    };
    let actual_level = match raw_level {
        serde_json::Value::String(value) => parse_prediction_level(value).ok_or_else(|| {
            DagMlError::CampaignValidation(format!(
                "candidate `{}` has invalid metric_level `{value}`",
                candidate.candidate_id
            ))
        })?,
        _ => {
            return Err(DagMlError::CampaignValidation(format!(
                "candidate `{}` metric_level must be a string",
                candidate.candidate_id
            )));
        }
    };
    if actual_level != required_level {
        return Err(DagMlError::CampaignValidation(format!(
            "candidate `{}` metric_level `{}` does not match required `{}`",
            candidate.candidate_id,
            prediction_level_name(actual_level),
            prediction_level_name(required_level)
        )));
    }
    Ok(())
}

fn parse_prediction_level(value: &str) -> Option<PredictionLevel> {
    match value {
        "observation" => Some(PredictionLevel::Observation),
        "sample" => Some(PredictionLevel::Sample),
        "target" => Some(PredictionLevel::Target),
        "group" => Some(PredictionLevel::Group),
        _ => None,
    }
}

fn prediction_level_name(level: PredictionLevel) -> &'static str {
    match level {
        PredictionLevel::Observation => "observation",
        PredictionLevel::Sample => "sample",
        PredictionLevel::Target => "target",
        PredictionLevel::Group => "group",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rmse_policy() -> SelectionPolicy {
        SelectionPolicy {
            id: "select:rmse".to_string(),
            metric: SelectionMetric {
                name: "rmse".to_string(),
                objective: MetricObjective::Minimize,
            },
            required_metric_level: None,
            require_finite: true,
        }
    }

    fn candidate(id: &str, rmse: f64) -> CandidateScore {
        CandidateScore {
            candidate_id: id.to_string(),
            metrics: BTreeMap::from([("rmse".to_string(), rmse)]),
            metadata: BTreeMap::new(),
        }
    }

    fn candidate_with_level(id: &str, rmse: f64, level: &str) -> CandidateScore {
        CandidateScore {
            candidate_id: id.to_string(),
            metrics: BTreeMap::from([("rmse".to_string(), rmse)]),
            metadata: BTreeMap::from([(
                "metric_level".to_string(),
                serde_json::Value::String(level.to_string()),
            )]),
        }
    }

    #[test]
    fn selects_lowest_metric_with_deterministic_tie_break() {
        let decision = select_candidate(
            &rmse_policy(),
            &[
                candidate("model:b", 1.0),
                candidate("model:a", 1.0),
                candidate("model:c", 2.0),
            ],
        )
        .unwrap();

        assert_eq!(decision.selected_candidate_id, "model:a");
        assert_eq!(decision.ranked_candidates[0].rank, 1);
    }

    #[test]
    fn grouped_selection_rejects_duplicate_candidate_ids() {
        assert!(select_candidate_groups(
            &rmse_policy(),
            &[candidate("model:a", 1.0), candidate("model:a", 2.0)],
            &BTreeMap::from([("branch:b0".to_string(), vec!["model:a".to_string()])]),
        )
        .is_err());
    }

    #[test]
    fn selection_policy_can_require_metric_level() {
        let mut policy = rmse_policy();
        policy.required_metric_level = Some(PredictionLevel::Sample);

        let decision = select_candidate(
            &policy,
            &[
                candidate_with_level("model:a", 1.0, "sample"),
                candidate_with_level("model:b", 2.0, "sample"),
            ],
        )
        .unwrap();
        assert_eq!(decision.selected_candidate_id, "model:a");
        assert_eq!(decision.metric_level, Some(PredictionLevel::Sample));

        assert!(select_candidate(
            &policy,
            &[
                candidate_with_level("model:a", 1.0, "sample"),
                candidate_with_level("model:b", 2.0, "target"),
            ],
        )
        .is_err());
        assert!(select_candidate(&policy, &[candidate("model:a", 1.0)]).is_err());
    }

    #[test]
    fn selects_sklearn_demo_branch_and_merge_variants() {
        let report: serde_json::Value = serde_json::from_str(include_str!(
            "../../../examples/generated/sklearn_complex_report.json"
        ))
        .unwrap();
        let branch_metrics = report["branch_variant_metrics"].as_object().unwrap();
        let candidates = branch_metrics
            .iter()
            .map(|(candidate_id, metrics)| CandidateScore {
                candidate_id: candidate_id.clone(),
                metrics: metrics
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(name, value)| (name.clone(), value.as_f64().unwrap()))
                    .collect(),
                metadata: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let groups = BTreeMap::from([
            (
                "branch:b0".to_string(),
                vec![
                    "branch:b0.variant:pca10_ridge_a03".to_string(),
                    "branch:b0.variant:pca16_ridge_a12".to_string(),
                ],
            ),
            (
                "branch:b1".to_string(),
                vec![
                    "branch:b1.variant:rf_select_k28".to_string(),
                    "branch:b1.variant:rf_select_k40".to_string(),
                ],
            ),
            (
                "branch:b2".to_string(),
                vec![
                    "branch:b2.variant:poly_extra_k45".to_string(),
                    "branch:b2.variant:poly_extra_k80".to_string(),
                ],
            ),
        ]);

        let decisions = select_candidate_groups(&rmse_policy(), &candidates, &groups).unwrap();
        assert_eq!(
            decisions["branch:b1"].selected_candidate_id,
            "branch:b1.variant:rf_select_k40"
        );

        let merge_metrics = report["merge_variant_metrics"].as_object().unwrap();
        let merge_candidates = merge_metrics
            .iter()
            .map(|(candidate_id, metrics)| CandidateScore {
                candidate_id: candidate_id.clone(),
                metrics: metrics
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(name, value)| (name.clone(), value.as_f64().unwrap()))
                    .collect(),
                metadata: BTreeMap::new(),
            })
            .collect::<Vec<_>>();
        let merge_decision = select_candidate(&rmse_policy(), &merge_candidates).unwrap();
        assert_eq!(
            merge_decision.selected_candidate_id,
            "merge:m1.pred_meta_original.meta:ridge"
        );
    }
}
