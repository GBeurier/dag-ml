use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::campaign::stable_json_fingerprint;
use crate::error::{DagMlError, Result};
use crate::ids::{NodeId, VariantId};
use crate::rng::SeedContext;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStrategy {
    #[default]
    None,
    Cartesian,
    Zip,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationChoice {
    pub label: String,
    pub value: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub param_overrides: Vec<GenerationParamOverride>,
    /// Operator-level variant: names the alternative sub-sequence this choice selects, distinct
    /// from a parameter variant (`param_overrides`). A choice with neither field is a value-only
    /// dimension; a choice may carry param_overrides XOR active_subsequence, never both. Skipped
    /// (None) when absent, so existing specs/fixtures stay byte-identical (Phase 2: no behavior).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_subsequence: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationParamOverride {
    pub node_id: NodeId,
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
}

impl GenerationChoice {
    fn validate(&self, dimension_name: &str) -> Result<()> {
        if self.label.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "generation dimension `{dimension_name}` has an empty choice label"
            )));
        }
        if !self.param_overrides.is_empty() && self.active_subsequence.is_some() {
            return Err(DagMlError::CampaignValidation(format!(
                "generation choice `{}` in dimension `{dimension_name}` cannot set both param_overrides and active_subsequence",
                self.label
            )));
        }
        if let Some(active_subsequence) = &self.active_subsequence {
            if active_subsequence.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "generation choice `{}` in dimension `{dimension_name}` has an empty active_subsequence",
                    self.label
                )));
            }
        }
        for override_spec in &self.param_overrides {
            override_spec.validate(dimension_name, &self.label)?;
        }
        Ok(())
    }
}

impl GenerationParamOverride {
    fn validate(&self, dimension_name: &str, choice_label: &str) -> Result<()> {
        if self.params.is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "generation choice `{choice_label}` in dimension `{dimension_name}` has an empty param override for node `{}`",
                self.node_id
            )));
        }
        for key in self.params.keys() {
            if key.trim().is_empty() {
                return Err(DagMlError::CampaignValidation(format!(
                    "generation choice `{choice_label}` in dimension `{dimension_name}` has an empty param override key for node `{}`",
                    self.node_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationDimension {
    pub name: String,
    #[serde(default)]
    pub choices: Vec<GenerationChoice>,
}

impl GenerationDimension {
    fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(
                "generation dimension name is empty".to_string(),
            ));
        }
        if self.choices.is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "generation dimension `{}` has no choices",
                self.name
            )));
        }
        let mut labels = BTreeSet::new();
        for choice in &self.choices {
            choice.validate(&self.name)?;
            if !labels.insert(choice.label.as_str()) {
                return Err(DagMlError::CampaignValidation(format!(
                    "generation dimension `{}` has duplicate choice `{}`",
                    self.name, choice.label
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationSpec {
    #[serde(default)]
    pub strategy: GenerationStrategy,
    #[serde(default)]
    pub dimensions: Vec<GenerationDimension>,
    #[serde(default)]
    pub max_variants: Option<usize>,
}

impl Default for GenerationSpec {
    fn default() -> Self {
        Self {
            strategy: GenerationStrategy::None,
            dimensions: Vec::new(),
            max_variants: Some(1),
        }
    }
}

impl GenerationSpec {
    pub fn validate(&self) -> Result<()> {
        if self.max_variants == Some(0) {
            return Err(DagMlError::CampaignValidation(
                "generation max_variants cannot be zero".to_string(),
            ));
        }
        if self.strategy == GenerationStrategy::None {
            if !self.dimensions.is_empty() {
                return Err(DagMlError::CampaignValidation(
                    "generation dimensions require cartesian or zip strategy".to_string(),
                ));
            }
            return Ok(());
        }

        if self.dimensions.is_empty() {
            return Err(DagMlError::CampaignValidation(
                "generation strategy requires at least one dimension".to_string(),
            ));
        }
        let mut names = BTreeSet::new();
        for dimension in &self.dimensions {
            dimension.validate()?;
            if !names.insert(dimension.name.as_str()) {
                return Err(DagMlError::CampaignValidation(format!(
                    "duplicate generation dimension `{}`",
                    dimension.name
                )));
            }
        }
        if self.strategy == GenerationStrategy::Zip {
            let expected = self.dimensions[0].choices.len();
            if self
                .dimensions
                .iter()
                .any(|dimension| dimension.choices.len() != expected)
            {
                return Err(DagMlError::CampaignValidation(
                    "zip generation requires every dimension to have the same number of choices"
                        .to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VariantPlan {
    pub variant_id: VariantId,
    #[serde(default)]
    pub choices: BTreeMap<String, GenerationChoice>,
    pub fingerprint: String,
    pub seed: Option<u64>,
}

impl VariantPlan {
    pub fn validate(&self) -> Result<()> {
        if self.fingerprint.trim().is_empty() {
            return Err(DagMlError::Planning(format!(
                "variant `{}` has an empty fingerprint",
                self.variant_id
            )));
        }
        for (dimension_name, choice) in &self.choices {
            choice.validate(dimension_name)?;
        }
        self.param_overrides_by_node()?;
        Ok(())
    }

    pub fn effective_params_for_node(
        &self,
        node_id: &NodeId,
        base_params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<BTreeMap<String, serde_json::Value>> {
        let overrides_by_node = self.param_overrides_by_node()?;
        let Some(overrides) = overrides_by_node.get(node_id) else {
            return Ok(base_params.clone());
        };
        let mut params = base_params.clone();
        params.extend(overrides.clone());
        Ok(params)
    }

    pub fn param_override_targets(&self) -> Result<BTreeSet<NodeId>> {
        Ok(self.param_overrides_by_node()?.into_keys().collect())
    }

    fn param_overrides_by_node(
        &self,
    ) -> Result<BTreeMap<NodeId, BTreeMap<String, serde_json::Value>>> {
        let mut overrides = BTreeMap::<NodeId, BTreeMap<String, serde_json::Value>>::new();
        let mut owners = BTreeMap::<(NodeId, String), String>::new();
        for (dimension_name, choice) in &self.choices {
            for override_spec in &choice.param_overrides {
                for (param_key, value) in &override_spec.params {
                    let owner_key = (override_spec.node_id.clone(), param_key.clone());
                    if let Some(previous) =
                        owners.insert(owner_key, format!("{dimension_name}:{}", choice.label))
                    {
                        return Err(DagMlError::CampaignValidation(format!(
                            "variant `{}` has conflicting generation overrides for `{}.{}` from `{previous}` and `{}:{}`",
                            self.variant_id,
                            override_spec.node_id,
                            param_key,
                            dimension_name,
                            choice.label
                        )));
                    }
                    overrides
                        .entry(override_spec.node_id.clone())
                        .or_default()
                        .insert(param_key.clone(), value.clone());
                }
            }
        }
        Ok(overrides)
    }
}

pub fn enumerate_variants(
    spec: &GenerationSpec,
    root_seed: Option<u64>,
) -> Result<Vec<VariantPlan>> {
    spec.validate()?;
    let mut variants = match spec.strategy {
        GenerationStrategy::None => vec![BTreeMap::new()],
        GenerationStrategy::Cartesian => cartesian_choices(&spec.dimensions),
        GenerationStrategy::Zip => zip_choices(&spec.dimensions),
    };
    if let Some(max_variants) = spec.max_variants {
        if variants.len() > max_variants {
            return Err(DagMlError::CampaignValidation(format!(
                "generation produced {} variants, above max_variants={max_variants}",
                variants.len()
            )));
        }
    }

    variants
        .drain(..)
        .map(|choices| variant_from_choices(choices, root_seed))
        .collect()
}

pub fn generation_spec_fingerprint(spec: &GenerationSpec) -> Result<String> {
    spec.validate()?;
    stable_json_fingerprint(spec)
}

fn cartesian_choices(
    dimensions: &[GenerationDimension],
) -> Vec<BTreeMap<String, GenerationChoice>> {
    let mut variants = vec![BTreeMap::new()];
    for dimension in dimensions {
        let mut next = Vec::with_capacity(variants.len() * dimension.choices.len());
        for existing in &variants {
            for choice in &dimension.choices {
                let mut merged = existing.clone();
                merged.insert(dimension.name.clone(), choice.clone());
                next.push(merged);
            }
        }
        variants = next;
    }
    variants
}

fn zip_choices(dimensions: &[GenerationDimension]) -> Vec<BTreeMap<String, GenerationChoice>> {
    let len = dimensions
        .first()
        .map_or(0, |dimension| dimension.choices.len());
    (0..len)
        .map(|idx| {
            dimensions
                .iter()
                .map(|dimension| (dimension.name.clone(), dimension.choices[idx].clone()))
                .collect::<BTreeMap<_, _>>()
        })
        .collect()
}

fn variant_from_choices(
    choices: BTreeMap<String, GenerationChoice>,
    root_seed: Option<u64>,
) -> Result<VariantPlan> {
    let fingerprint = stable_json_fingerprint(&choices)?;
    let suffix = if choices.is_empty() {
        "base".to_string()
    } else {
        fingerprint[..16].to_string()
    };
    let variant_id = VariantId::new(format!("variant:{suffix}"))?;
    let seed = root_seed.map(|seed| {
        SeedContext::root(seed)
            .child(format!("variant:{variant_id}"))
            .derive_u64("variant")
    });
    let variant = VariantPlan {
        variant_id,
        choices,
        fingerprint,
        seed,
    };
    variant.validate()?;
    Ok(variant)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn choice(label: &str, value: serde_json::Value) -> GenerationChoice {
        GenerationChoice {
            label: label.to_string(),
            value,
            param_overrides: Vec::new(),
            active_subsequence: None,
        }
    }

    fn override_choice(
        label: &str,
        node_id: &str,
        params: BTreeMap<String, serde_json::Value>,
    ) -> GenerationChoice {
        GenerationChoice {
            label: label.to_string(),
            value: json!(label),
            param_overrides: vec![GenerationParamOverride {
                node_id: NodeId::new(node_id).unwrap(),
                params,
            }],
            active_subsequence: None,
        }
    }

    #[test]
    fn default_generation_produces_base_variant() {
        let variants = enumerate_variants(&GenerationSpec::default(), Some(7)).unwrap();

        assert_eq!(variants.len(), 1);
        assert_eq!(variants[0].variant_id.as_str(), "variant:base");
        assert!(variants[0].choices.is_empty());
        assert!(variants[0].seed.is_some());
    }

    #[test]
    fn cartesian_generation_is_deterministic_and_fingerprinted() {
        let spec = GenerationSpec {
            strategy: GenerationStrategy::Cartesian,
            dimensions: vec![
                GenerationDimension {
                    name: "model".to_string(),
                    choices: vec![choice("pls", json!("pls")), choice("rf", json!("rf"))],
                },
                GenerationDimension {
                    name: "window".to_string(),
                    choices: vec![choice("short", json!(7)), choice("long", json!(21))],
                },
            ],
            max_variants: Some(4),
        };

        let left = enumerate_variants(&spec, Some(11)).unwrap();
        let right = enumerate_variants(&spec, Some(11)).unwrap();

        assert_eq!(left.len(), 4);
        assert_eq!(left, right);
        let fingerprint = generation_spec_fingerprint(&spec).unwrap();
        let mut changed_spec = spec.clone();
        changed_spec.dimensions[0].choices[0].value = json!("changed");
        assert_eq!(fingerprint, generation_spec_fingerprint(&spec).unwrap());
        assert_ne!(
            fingerprint,
            generation_spec_fingerprint(&changed_spec).unwrap()
        );
        assert_ne!(left[0].variant_id, left[1].variant_id);
        assert_eq!(left[0].choices["model"].label, "pls");
        assert_eq!(left[0].choices["window"].label, "short");
    }

    /// Phase 2 byte-identity gate for the additive `active_subsequence` / `variant_label` fields.
    ///
    /// The committed example specs/fixtures carry NEITHER new field, so `skip_serializing_if =
    /// "Option::is_none"` must keep them invisible: the structs round-trip to byte-identical JSON
    /// and produce byte-identical fingerprints (`generation_spec_fingerprint` /
    /// `stable_json_fingerprint`) as before the struct change. The fingerprints are pinned to the
    /// values produced before the fields existed, proving pure additivity / zero behavior change.
    #[test]
    fn additive_variant_fields_are_invisible_when_absent() {
        // `examples/campaign_oof_generation.json` — a CampaignSpec whose `generation` block is a
        // value-only cartesian search (no param_overrides, no active_subsequence on any choice).
        let campaign: crate::plan::CampaignSpec = serde_json::from_str(include_str!(
            "../../../examples/campaign_oof_generation.json"
        ))
        .unwrap();
        let generation_serialized = serde_json::to_string(&campaign.generation).unwrap();
        assert!(
            !generation_serialized.contains("active_subsequence"),
            "absent active_subsequence must not serialize: {generation_serialized}"
        );
        // Fingerprint pinned to the pre-field value: unchanged proves the new field is invisible.
        assert_eq!(
            generation_spec_fingerprint(&campaign.generation).unwrap(),
            "8d10bce07876d936ab6a62f13063a8d241c967a1578b6d2295a43c26275edf47"
        );

        // `examples/fixtures/score_set.json` — a ScoreSet whose reports carry no variant_label
        // (one report carries a variant_id, none carry variant_label).
        let score_set: crate::metrics::ScoreSet =
            serde_json::from_str(include_str!("../../../examples/fixtures/score_set.json"))
                .unwrap();
        let score_set_serialized = serde_json::to_string(&score_set).unwrap();
        assert!(
            !score_set_serialized.contains("variant_label"),
            "absent variant_label must not serialize: {score_set_serialized}"
        );
        assert_eq!(
            stable_json_fingerprint(&score_set).unwrap(),
            "e99fa78d79ef2a2b99927276cfaf4c265210abf3cf8b3575477355264fda4a9d"
        );
    }

    #[test]
    fn choice_cannot_set_both_param_overrides_and_active_subsequence() {
        let choice = GenerationChoice {
            label: "both".to_string(),
            value: json!("both"),
            param_overrides: vec![GenerationParamOverride {
                node_id: NodeId::new("model:base").unwrap(),
                params: BTreeMap::from([("n_components".to_string(), json!(4))]),
            }],
            active_subsequence: Some("alt".to_string()),
        };
        let error = choice.validate("dim").unwrap_err().to_string();
        assert!(
            error.contains("cannot set both param_overrides and active_subsequence"),
            "{error}"
        );

        // Only-param_overrides (existing param variant) stays legal.
        let param_only = GenerationChoice {
            active_subsequence: None,
            ..choice.clone()
        };
        param_only.validate("dim").unwrap();

        // Only-active_subsequence (operator variant) stays legal.
        let operator_only = GenerationChoice {
            param_overrides: Vec::new(),
            ..choice.clone()
        };
        operator_only.validate("dim").unwrap();

        // Neither (value-only) stays legal.
        let value_only = GenerationChoice {
            param_overrides: Vec::new(),
            active_subsequence: None,
            ..choice
        };
        value_only.validate("dim").unwrap();
    }

    #[test]
    fn choice_rejects_empty_active_subsequence() {
        // The schema (minLength:1) and both Python validator paths require a non-empty
        // active_subsequence; the Rust validate must reject empty/whitespace identically so
        // CampaignSpec/ExecutionPlan validation does not accept a contract shape they reject.
        for blank in ["", "   "] {
            let choice = GenerationChoice {
                label: "op".to_string(),
                value: json!("op"),
                param_overrides: Vec::new(),
                active_subsequence: Some(blank.to_string()),
            };
            let error = choice.validate("dim").unwrap_err().to_string();
            assert!(error.contains("has an empty active_subsequence"), "{error}");
        }
    }

    #[test]
    fn zip_generation_requires_same_choice_count() {
        let spec = GenerationSpec {
            strategy: GenerationStrategy::Zip,
            dimensions: vec![
                GenerationDimension {
                    name: "a".to_string(),
                    choices: vec![choice("a1", json!(1))],
                },
                GenerationDimension {
                    name: "b".to_string(),
                    choices: vec![choice("b1", json!(1)), choice("b2", json!(2))],
                },
            ],
            max_variants: None,
        };

        assert!(spec.validate().is_err());
    }

    #[test]
    fn generation_respects_variant_limit() {
        let spec = GenerationSpec {
            strategy: GenerationStrategy::Cartesian,
            dimensions: vec![GenerationDimension {
                name: "x".to_string(),
                choices: vec![choice("a", json!(1)), choice("b", json!(2))],
            }],
            max_variants: Some(1),
        };

        assert!(enumerate_variants(&spec, None).is_err());
    }

    #[test]
    fn variant_applies_node_param_overrides() {
        let spec = GenerationSpec {
            strategy: GenerationStrategy::Cartesian,
            dimensions: vec![GenerationDimension {
                name: "model_family".to_string(),
                choices: vec![override_choice(
                    "pls",
                    "model:base",
                    BTreeMap::from([("n_components".to_string(), json!(8))]),
                )],
            }],
            max_variants: Some(1),
        };
        let variants = enumerate_variants(&spec, Some(7)).unwrap();
        let base = BTreeMap::from([("scale".to_string(), json!(true))]);

        let params = variants[0]
            .effective_params_for_node(&NodeId::new("model:base").unwrap(), &base)
            .unwrap();

        assert_eq!(params["scale"], json!(true));
        assert_eq!(params["n_components"], json!(8));
    }

    #[test]
    fn variant_rejects_conflicting_param_overrides() {
        let spec = GenerationSpec {
            strategy: GenerationStrategy::Cartesian,
            dimensions: vec![
                GenerationDimension {
                    name: "family".to_string(),
                    choices: vec![override_choice(
                        "pls",
                        "model:base",
                        BTreeMap::from([("alpha".to_string(), json!(1))]),
                    )],
                },
                GenerationDimension {
                    name: "regularization".to_string(),
                    choices: vec![override_choice(
                        "ridge",
                        "model:base",
                        BTreeMap::from([("alpha".to_string(), json!(2))]),
                    )],
                },
            ],
            max_variants: Some(1),
        };

        let error = enumerate_variants(&spec, None).unwrap_err().to_string();

        assert!(error.contains("conflicting generation overrides"));
    }
}
