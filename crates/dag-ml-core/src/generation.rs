use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::campaign::stable_json_fingerprint;
use crate::error::{DagMlError, Result};
use crate::ids::VariantId;
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
}

impl GenerationChoice {
    fn validate(&self, dimension_name: &str) -> Result<()> {
        if self.label.trim().is_empty() {
            return Err(DagMlError::CampaignValidation(format!(
                "generation dimension `{dimension_name}` has an empty choice label"
            )));
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
        Ok(())
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
    Ok(VariantPlan {
        variant_id,
        choices,
        fingerprint,
        seed,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn choice(label: &str, value: serde_json::Value) -> GenerationChoice {
        GenerationChoice {
            label: label.to_string(),
            value,
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
        assert_ne!(left[0].variant_id, left[1].variant_id);
        assert_eq!(left[0].choices["model"].label, "pls");
        assert_eq!(left[0].choices["window"].label, "short");
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
}
