//! Generator compilation and expansion: variant/param-generator dimensions,
//! range/log-range/grid/pick/arrange generators, grid-row/combination/
//! permutation builders, and generated-sequence namespacing.

use super::*;

#[derive(Clone, Debug)]
pub(crate) struct GeneratedSequence {
    pub(crate) id: String,
    pub(crate) labels: Vec<String>,
    pub(crate) steps: Vec<PipelineDslStep>,
    pub(crate) metadata: BTreeMap<String, serde_json::Value>,
}

pub(crate) fn compile_explicit_generation_dimensions(
    dimensions: &[PipelineDslGenerationDimension],
    nodes: &[NodeSpec],
) -> Result<Vec<GenerationDimension>> {
    if dimensions.is_empty() {
        return Ok(Vec::new());
    }
    let node_ids = nodes
        .iter()
        .map(|node| node.id.clone())
        .collect::<BTreeSet<_>>();
    dimensions
        .iter()
        .map(|dimension| compile_explicit_generation_dimension(dimension, &node_ids))
        .collect()
}
pub(crate) fn compile_explicit_generation_dimension(
    dimension: &PipelineDslGenerationDimension,
    node_ids: &BTreeSet<NodeId>,
) -> Result<GenerationDimension> {
    let choices = dimension
        .choices
        .iter()
        .map(|choice| compile_explicit_generation_choice(&dimension.name, choice, node_ids))
        .collect::<Result<Vec<_>>>()?;
    Ok(GenerationDimension {
        name: dimension.name.clone(),
        choices,
    })
}
pub(crate) fn compile_explicit_generation_choice(
    dimension_name: &str,
    choice: &PipelineDslGenerationChoice,
    node_ids: &BTreeSet<NodeId>,
) -> Result<GenerationChoice> {
    if choice.param_overrides.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generation choice `{}` in dimension `{dimension_name}` has no param_overrides",
            choice.label
        )));
    }
    let param_overrides = choice
        .param_overrides
        .iter()
        .map(|override_spec| {
            if !node_ids.contains(&override_spec.node_id) {
                return Err(DagMlError::GraphValidation(format!(
                    "pipeline DSL generation choice `{}` in dimension `{dimension_name}` references unknown node `{}`",
                    choice.label, override_spec.node_id
                )));
            }
            Ok(GenerationParamOverride {
                node_id: override_spec.node_id.clone(),
                params: override_spec.params.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let value = match &choice.value {
        Some(value) => value.clone(),
        None => explicit_generation_choice_value(&param_overrides)?,
    };
    Ok(GenerationChoice {
        label: choice.label.clone(),
        value,
        param_overrides,
    })
}
pub(crate) fn explicit_generation_choice_value(
    param_overrides: &[GenerationParamOverride],
) -> Result<serde_json::Value> {
    let mut by_node = serde_json::Map::new();
    for override_spec in param_overrides {
        let value = serde_json::to_value(&override_spec.params).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize DSL generation override for node `{}`: {error}",
                override_spec.node_id
            ))
        })?;
        by_node.insert(override_spec.node_id.to_string(), value);
    }
    Ok(serde_json::Value::Object(by_node))
}
pub(crate) fn compile_variant_choice_dimension(
    node_id: &NodeId,
    choices: &[PipelineDslVariantChoice],
) -> Result<GenerationDimension> {
    Ok(GenerationDimension {
        name: format!("{node_id}.params"),
        choices: choices
            .iter()
            .map(|choice| {
                if choice.params.is_empty() {
                    return Err(DagMlError::GraphValidation(format!(
                        "pipeline DSL variant `{}` for node `{node_id}` has no params",
                        choice.label
                    )));
                }
                let value = match &choice.value {
                    Some(value) => value.clone(),
                    None => serde_json::to_value(&choice.params).map_err(|error| {
                        DagMlError::GraphValidation(format!(
                            "failed to serialize pipeline DSL variant `{}` for node `{node_id}`: {error}",
                            choice.label
                        ))
                    })?,
                };
                Ok(GenerationChoice {
                    label: choice.label.clone(),
                    value,
                    param_overrides: vec![GenerationParamOverride {
                        node_id: node_id.clone(),
                        params: choice.params.clone(),
                    }],
                })
            })
            .collect::<Result<Vec<_>>>()?,
    })
}
pub(crate) fn compile_param_generator_dimension(
    node_id: &NodeId,
    generator: &PipelineDslParamGenerator,
) -> Result<GenerationDimension> {
    match generator {
        PipelineDslParamGenerator::Or {
            name,
            param,
            values,
            count,
        } => compile_or_generator(node_id, name.as_deref(), param, values, *count),
        PipelineDslParamGenerator::Range {
            name,
            param,
            start,
            stop,
            step,
            inclusive,
            count,
        } => compile_range_generator(RangeGeneratorSpec {
            node_id,
            name: name.as_deref(),
            param,
            start: *start,
            stop: *stop,
            step: *step,
            inclusive: *inclusive,
            count: *count,
        }),
        PipelineDslParamGenerator::LogRange {
            name,
            param,
            start,
            stop,
            count,
            base,
        } => compile_log_range_generator(
            node_id,
            name.as_deref(),
            param,
            *start,
            *stop,
            *count,
            *base,
        ),
        PipelineDslParamGenerator::Grid {
            name,
            params,
            count,
        } => compile_grid_generator(node_id, name.as_deref(), params, *count),
        PipelineDslParamGenerator::Pick {
            name,
            param,
            values,
            sizes,
            count,
        } => compile_pick_arrange_generator(
            node_id,
            name.as_deref(),
            param,
            values,
            sizes,
            *count,
            PickArrangeMode::Pick,
        ),
        PipelineDslParamGenerator::Arrange {
            name,
            param,
            values,
            sizes,
            count,
        } => compile_pick_arrange_generator(
            node_id,
            name.as_deref(),
            param,
            values,
            sizes,
            *count,
            PickArrangeMode::Arrange,
        ),
    }
}
pub(crate) fn compile_or_generator(
    node_id: &NodeId,
    name: Option<&str>,
    param: &str,
    values: &[PipelineDslGeneratorValue],
    count: Option<usize>,
) -> Result<GenerationDimension> {
    validate_param_name(node_id, param)?;
    validate_count(node_id, name, count)?;
    if values.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` for node `{node_id}` has no values",
            generator_dimension_name(node_id, name, Some(param), "or")
        )));
    }
    let mut choices = values
        .iter()
        .enumerate()
        .map(|(index, value)| single_param_generation_choice(node_id, param, index, value))
        .collect::<Result<Vec<_>>>()?;
    apply_choice_count(&mut choices, count);
    Ok(GenerationDimension {
        name: generator_dimension_name(node_id, name, Some(param), "or"),
        choices,
    })
}
pub(crate) struct RangeGeneratorSpec<'a> {
    node_id: &'a NodeId,
    name: Option<&'a str>,
    param: &'a str,
    start: f64,
    stop: f64,
    step: f64,
    inclusive: bool,
    count: Option<usize>,
}
pub(crate) fn compile_range_generator(spec: RangeGeneratorSpec<'_>) -> Result<GenerationDimension> {
    validate_param_name(spec.node_id, spec.param)?;
    validate_count(spec.node_id, spec.name, spec.count)?;
    validate_finite(spec.node_id, spec.param, "range start", spec.start)?;
    validate_finite(spec.node_id, spec.param, "range stop", spec.stop)?;
    validate_finite(spec.node_id, spec.param, "range step", spec.step)?;
    if spec.step == 0.0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL range generator for `{}.{}` has zero step",
            spec.node_id, spec.param
        )));
    }
    if spec.start < spec.stop && spec.step < 0.0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL range generator for `{}.{}` steps away from stop",
            spec.node_id, spec.param
        )));
    }
    if spec.start > spec.stop && spec.step > 0.0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL range generator for `{}.{}` steps away from stop",
            spec.node_id, spec.param
        )));
    }
    let mut values = Vec::new();
    let mut current = spec.start;
    let mut guard = 0usize;
    while range_contains(current, spec.stop, spec.step, spec.inclusive) {
        values.push(json_number(current, spec.node_id, spec.param)?);
        current += spec.step;
        guard += 1;
        if guard > 10_000 {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL range generator for `{}.{}` produced more than 10000 values",
                spec.node_id, spec.param
            )));
        }
    }
    if values.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL range generator for `{}.{}` produced no values",
            spec.node_id, spec.param
        )));
    }
    let wrapped = values
        .into_iter()
        .map(PipelineDslGeneratorValue::Value)
        .collect::<Vec<_>>();
    compile_or_generator(spec.node_id, spec.name, spec.param, &wrapped, spec.count).map(
        |mut dimension| {
            dimension.name =
                generator_dimension_name(spec.node_id, spec.name, Some(spec.param), "range");
            dimension
        },
    )
}
pub(crate) fn compile_log_range_generator(
    node_id: &NodeId,
    name: Option<&str>,
    param: &str,
    start: f64,
    stop: f64,
    count: usize,
    base: f64,
) -> Result<GenerationDimension> {
    validate_param_name(node_id, param)?;
    validate_finite(node_id, param, "log_range start", start)?;
    validate_finite(node_id, param, "log_range stop", stop)?;
    validate_finite(node_id, param, "log_range base", base)?;
    if start <= 0.0 || stop <= 0.0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL log_range generator for `{node_id}.{param}` requires positive start and stop"
        )));
    }
    if count == 0 {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL log_range generator for `{node_id}.{param}` has count=0"
        )));
    }
    if base <= 0.0 || (base - 1.0).abs() < f64::EPSILON {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL log_range generator for `{node_id}.{param}` requires base > 0 and != 1"
        )));
    }
    let start_log = start.log(base);
    let stop_log = stop.log(base);
    let values = if count == 1 {
        vec![json_number(start, node_id, param)?]
    } else {
        (0..count)
            .map(|index| {
                let ratio = index as f64 / (count - 1) as f64;
                json_number(
                    base.powf(start_log + (stop_log - start_log) * ratio),
                    node_id,
                    param,
                )
            })
            .collect::<Result<Vec<_>>>()?
    };
    let wrapped = values
        .into_iter()
        .map(PipelineDslGeneratorValue::Value)
        .collect::<Vec<_>>();
    compile_or_generator(node_id, name, param, &wrapped, None).map(|mut dimension| {
        dimension.name = generator_dimension_name(node_id, name, Some(param), "log_range");
        dimension
    })
}
pub(crate) fn compile_grid_generator(
    node_id: &NodeId,
    name: Option<&str>,
    params: &BTreeMap<String, Vec<PipelineDslGeneratorValue>>,
    count: Option<usize>,
) -> Result<GenerationDimension> {
    validate_count(node_id, name, count)?;
    if params.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL grid generator for node `{node_id}` has no params"
        )));
    }
    for (param, values) in params {
        validate_param_name(node_id, param)?;
        if values.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL grid generator for `{node_id}.{param}` has no values"
            )));
        }
    }
    let entries = params
        .iter()
        .map(|(param, values)| (param.as_str(), values.as_slice()))
        .collect::<Vec<_>>();
    let mut rows = Vec::<BTreeMap<String, PipelineDslGeneratorValue>>::new();
    build_grid_rows(&entries, 0, &mut BTreeMap::new(), &mut rows, count);
    let choices = rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| multi_param_generation_choice(node_id, index, row))
        .collect::<Result<Vec<_>>>()?;
    Ok(GenerationDimension {
        name: generator_dimension_name(node_id, name, None, "grid"),
        choices,
    })
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PickArrangeMode {
    Pick,
    Arrange,
}
pub(crate) fn compile_pick_arrange_generator(
    node_id: &NodeId,
    name: Option<&str>,
    param: &str,
    values: &[PipelineDslGeneratorValue],
    sizes: &[usize],
    count: Option<usize>,
    mode: PickArrangeMode,
) -> Result<GenerationDimension> {
    validate_param_name(node_id, param)?;
    validate_count(node_id, name, count)?;
    if values.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL {:?} generator for `{node_id}.{param}` has no values",
            mode
        )));
    }
    if sizes.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL {:?} generator for `{node_id}.{param}` has no sizes",
            mode
        )));
    }
    let mut selections = Vec::<Vec<usize>>::new();
    for size in sizes {
        if *size == 0 || *size > values.len() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL {:?} generator for `{node_id}.{param}` has invalid size `{size}`",
                mode
            )));
        }
        match mode {
            PickArrangeMode::Pick => build_combinations(
                values.len(),
                *size,
                0,
                &mut Vec::new(),
                &mut selections,
                count,
            ),
            PickArrangeMode::Arrange => build_permutations(
                values.len(),
                *size,
                &mut BTreeSet::new(),
                &mut Vec::new(),
                &mut selections,
                count,
            ),
        }
        if count.is_some_and(|limit| selections.len() >= limit) {
            break;
        }
    }
    let mut choices = selections
        .into_iter()
        .enumerate()
        .map(|(index, selection)| {
            let selected_values = selection
                .iter()
                .map(|selected| values[*selected].value().clone())
                .collect::<Vec<_>>();
            let selected_labels = selection
                .iter()
                .map(|selected| values[*selected].label_fragment())
                .collect::<Vec<_>>();
            let mut params = BTreeMap::new();
            params.insert(param.to_string(), serde_json::Value::Array(selected_values));
            Ok(GenerationChoice {
                label: format!(
                    "{index:04}_{}_{}",
                    match mode {
                        PickArrangeMode::Pick => "pick",
                        PickArrangeMode::Arrange => "arrange",
                    },
                    sanitize_generation_label(&selected_labels.join("_"))
                ),
                value: serde_json::to_value(&params).map_err(|error| {
                    DagMlError::GraphValidation(format!(
                        "failed to serialize pipeline DSL {:?} generator choice for `{node_id}.{param}`: {error}",
                        mode
                    ))
                })?,
                param_overrides: vec![GenerationParamOverride {
                    node_id: node_id.clone(),
                    params,
                }],
            })
        })
        .collect::<Result<Vec<_>>>()?;
    apply_choice_count(&mut choices, count);
    Ok(GenerationDimension {
        name: generator_dimension_name(
            node_id,
            name,
            Some(param),
            match mode {
                PickArrangeMode::Pick => "pick",
                PickArrangeMode::Arrange => "arrange",
            },
        ),
        choices,
    })
}
pub(crate) fn single_param_generation_choice(
    node_id: &NodeId,
    param: &str,
    index: usize,
    value: &PipelineDslGeneratorValue,
) -> Result<GenerationChoice> {
    let mut params = BTreeMap::new();
    params.insert(param.to_string(), value.value().clone());
    Ok(GenerationChoice {
        label: format!(
            "{index:04}_{}_{}",
            sanitize_generation_label(param),
            value.label_fragment()
        ),
        value: serde_json::to_value(&params).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL generator choice for `{node_id}.{param}`: {error}"
            ))
        })?,
        param_overrides: vec![GenerationParamOverride {
            node_id: node_id.clone(),
            params,
        }],
    })
}
pub(crate) fn multi_param_generation_choice(
    node_id: &NodeId,
    index: usize,
    row: BTreeMap<String, PipelineDslGeneratorValue>,
) -> Result<GenerationChoice> {
    let mut params = BTreeMap::new();
    let mut label_parts = Vec::new();
    for (param, value) in row {
        label_parts.push(format!(
            "{}_{}",
            sanitize_generation_label(&param),
            value.label_fragment()
        ));
        params.insert(param, value.value().clone());
    }
    Ok(GenerationChoice {
        label: format!("{index:04}_{}", label_parts.join("__")),
        value: serde_json::to_value(&params).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL grid generator choice for node `{node_id}`: {error}"
            ))
        })?,
        param_overrides: vec![GenerationParamOverride {
            node_id: node_id.clone(),
            params,
        }],
    })
}
pub(crate) fn build_grid_rows(
    entries: &[(&str, &[PipelineDslGeneratorValue])],
    entry_index: usize,
    current: &mut BTreeMap<String, PipelineDslGeneratorValue>,
    rows: &mut Vec<BTreeMap<String, PipelineDslGeneratorValue>>,
    count: Option<usize>,
) {
    if count.is_some_and(|limit| rows.len() >= limit) {
        return;
    }
    if entry_index == entries.len() {
        rows.push(current.clone());
        return;
    }
    let (param, values) = entries[entry_index];
    for value in values {
        current.insert(param.to_string(), value.clone());
        build_grid_rows(entries, entry_index + 1, current, rows, count);
        current.remove(param);
        if count.is_some_and(|limit| rows.len() >= limit) {
            break;
        }
    }
}
pub(crate) fn build_combinations(
    value_count: usize,
    size: usize,
    start: usize,
    current: &mut Vec<usize>,
    selections: &mut Vec<Vec<usize>>,
    count: Option<usize>,
) {
    if count.is_some_and(|limit| selections.len() >= limit) {
        return;
    }
    if current.len() == size {
        selections.push(current.clone());
        return;
    }
    let remaining = size - current.len();
    if value_count < remaining {
        return;
    }
    for index in start..=value_count - remaining {
        current.push(index);
        build_combinations(value_count, size, index + 1, current, selections, count);
        current.pop();
        if count.is_some_and(|limit| selections.len() >= limit) {
            break;
        }
    }
}
pub(crate) fn build_permutations(
    value_count: usize,
    size: usize,
    used: &mut BTreeSet<usize>,
    current: &mut Vec<usize>,
    selections: &mut Vec<Vec<usize>>,
    count: Option<usize>,
) {
    if count.is_some_and(|limit| selections.len() >= limit) {
        return;
    }
    if current.len() == size {
        selections.push(current.clone());
        return;
    }
    for index in 0..value_count {
        if used.contains(&index) {
            continue;
        }
        used.insert(index);
        current.push(index);
        build_permutations(value_count, size, used, current, selections, count);
        current.pop();
        used.remove(&index);
        if count.is_some_and(|limit| selections.len() >= limit) {
            break;
        }
    }
}
pub(crate) fn apply_choice_count(choices: &mut Vec<GenerationChoice>, count: Option<usize>) {
    if let Some(limit) = count {
        choices.truncate(limit);
    }
}
pub(crate) fn validate_count(
    node_id: &NodeId,
    name: Option<&str>,
    count: Option<usize>,
) -> Result<()> {
    if count == Some(0) {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` for node `{node_id}` has count=0",
            generator_dimension_name(node_id, name, None, "params")
        )));
    }
    Ok(())
}
pub(crate) fn validate_param_name(node_id: &NodeId, param: &str) -> Result<()> {
    if param.trim().is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL param generator for node `{node_id}` has an empty param name"
        )));
    }
    Ok(())
}
pub(crate) fn validate_finite(
    node_id: &NodeId,
    param: &str,
    field: &str,
    value: f64,
) -> Result<()> {
    if !value.is_finite() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL {field} for `{node_id}.{param}` must be finite"
        )));
    }
    Ok(())
}
pub(crate) fn range_contains(current: f64, stop: f64, step: f64, inclusive: bool) -> bool {
    let epsilon = step.abs() * 1e-12 + f64::EPSILON;
    if step > 0.0 {
        if inclusive {
            current <= stop + epsilon
        } else {
            current < stop - epsilon
        }
    } else if inclusive {
        current >= stop - epsilon
    } else {
        current > stop + epsilon
    }
}
pub(crate) fn json_number(value: f64, node_id: &NodeId, param: &str) -> Result<serde_json::Value> {
    let number = serde_json::Number::from_f64(value).ok_or_else(|| {
        DagMlError::GraphValidation(format!(
            "pipeline DSL numeric generator for `{node_id}.{param}` produced a non-finite value"
        ))
    })?;
    Ok(serde_json::Value::Number(canonical_generator_number(
        number,
    )))
}
/// Normalize a generated numeric value to its JSON round-trip *fixpoint*.
///
/// `serde_json`'s shortest-decimal float formatting is not always a round-trip
/// fixpoint: a value such as `0.010000000000000005` renders to that text, but
/// re-parsing the text and re-rendering yields `0.010000000000000004`. Generator
/// choices store the value (fingerprinted by the campaign generation spec, which
/// is re-parsed from JSON at plan time) and derive the choice label from the same
/// value at compile time (kept verbatim as a string in the graph that backs the
/// `search_space_fingerprint`). Without normalization the value drifts by one ULP
/// of decimal text across the compile→serialize→plan JSON round-trip while the
/// label string does not, so the two fingerprints disagree and planning rejects
/// the plan (`search_space_fingerprint does not match campaign generation spec`).
///
/// One parse→reserialize→parse pass reaches the fixpoint (verified: a second pass
/// is a no-op for every value), so canonicalizing here — at the single point that
/// produces every numeric generator value — makes the value its own round-trip
/// fixpoint. Both the stored value and the derived label then render identically
/// no matter how many JSON round-trips the artifact takes, so the graph-side and
/// campaign-side fingerprints agree. Integer-valued and already-stable decimals
/// (every `range`/`grid` value in practice) are fixpoints already, so this is a
/// no-op for them.
pub(crate) fn canonical_generator_number(number: serde_json::Number) -> serde_json::Number {
    let rendered = number.to_string();
    serde_json::from_str::<serde_json::Number>(&rendered).unwrap_or(number)
}
pub(crate) fn generator_dimension_name(
    node_id: &NodeId,
    name: Option<&str>,
    param: Option<&str>,
    suffix: &str,
) -> String {
    if let Some(name) = name {
        return name.to_string();
    }
    match param {
        Some(param) => format!("{node_id}.{param}.{suffix}"),
        None => format!("{node_id}.{suffix}"),
    }
}
impl PipelineDslGeneratorValue {
    fn value(&self) -> &serde_json::Value {
        match self {
            Self::Labeled { value, .. } | Self::Value(value) => value,
        }
    }

    fn label_fragment(&self) -> String {
        match self {
            Self::Labeled { label, .. } => sanitize_generation_label(label),
            Self::Value(value) => {
                let rendered = match value {
                    serde_json::Value::String(value) => value.clone(),
                    _ => serde_json::to_string(value).unwrap_or_else(|_| "value".to_string()),
                };
                sanitize_generation_label(&rendered)
            }
        }
    }
}
pub(crate) fn sanitize_generation_label(input: &str) -> String {
    let sanitized = input
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '-' | '.') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if sanitized.is_empty() {
        "value".to_string()
    } else {
        sanitized
    }
}
pub(crate) fn build_generation_spec(
    requested_strategy: Option<GenerationStrategy>,
    max_variants: Option<usize>,
    dimensions: Vec<GenerationDimension>,
) -> Result<GenerationSpec> {
    let strategy = requested_strategy.unwrap_or(if dimensions.is_empty() {
        GenerationStrategy::None
    } else {
        GenerationStrategy::Cartesian
    });
    let generation = GenerationSpec {
        strategy,
        dimensions,
        max_variants: if strategy == GenerationStrategy::None {
            Some(1)
        } else {
            max_variants
        },
    };
    generation.validate()?;
    Ok(generation)
}
pub(crate) fn expand_generator_sequences(
    step: &PipelineDslGeneratorStep,
) -> Result<Vec<GeneratedSequence>> {
    if step.count == Some(0) {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` count cannot be zero",
            step.id
        )));
    }
    match step.mode {
        PipelineDslGeneratorMode::Or => expand_or_generator_sequences(step),
        PipelineDslGeneratorMode::Cartesian => expand_cartesian_generator_sequences(step),
    }
}
pub(crate) fn expand_or_generator_sequences(
    step: &PipelineDslGeneratorStep,
) -> Result<Vec<GeneratedSequence>> {
    if !step.stages.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` uses mode `or` but declares cartesian stages",
            step.id
        )));
    }
    if step.branches.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` has no branches",
            step.id
        )));
    }
    let options = step
        .branches
        .iter()
        .enumerate()
        .map(|(index, branch)| {
            validate_branch_id(&branch.id)?;
            Ok(GeneratedSequence {
                id: generator_choice_id(&step.id, index),
                labels: vec![branch.id.clone()],
                steps: branch.steps.clone(),
                metadata: branch.metadata.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let choices = if let Some(sizes) = selection_sizes(step.pick)? {
        generated_pick_sequences(&options, &step.id, "pick", &sizes, step.count)?
    } else if let Some(sizes) = selection_sizes(step.arrange)? {
        generated_arrange_sequences(&options, &step.id, "arrange", &sizes, step.count)?
    } else {
        truncate_generated_sequences(options, step.count)
    };

    let choices = if let Some(sizes) = selection_sizes(step.then_pick)? {
        generated_pick_sequences(&choices, &step.id, "then_pick", &sizes, step.count)?
    } else if let Some(sizes) = selection_sizes(step.then_arrange)? {
        generated_arrange_sequences(&choices, &step.id, "then_arrange", &sizes, step.count)?
    } else {
        choices
    };
    Ok(truncate_generated_sequences(choices, step.count))
}
pub(crate) fn expand_cartesian_generator_sequences(
    step: &PipelineDslGeneratorStep,
) -> Result<Vec<GeneratedSequence>> {
    if !step.branches.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` uses mode `cartesian` but declares direct branches",
            step.id
        )));
    }
    if step.stages.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` has no cartesian stages",
            step.id
        )));
    }
    if step.pick.is_some()
        || step.arrange.is_some()
        || step.then_pick.is_some()
        || step.then_arrange.is_some()
    {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` cannot combine cartesian mode with pick/arrange selectors",
            step.id
        )));
    }

    let mut stage_options = Vec::<Vec<GeneratedSequence>>::new();
    for (stage_index, stage) in step.stages.iter().enumerate() {
        validate_branch_id(&stage.id)?;
        if stage.branches.is_empty() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL generator `{}` stage `{}` has no branches",
                step.id, stage.id
            )));
        }
        let mut options = Vec::new();
        for branch in &stage.branches {
            validate_branch_id(&branch.id)?;
            let mut metadata = branch.metadata.clone();
            if let Some(selector) = &stage.selector {
                metadata.insert("dsl_generator_stage_selector".to_string(), selector.clone());
            }
            if !stage.metadata.is_empty() {
                metadata.insert(
                    "dsl_generator_stage_metadata".to_string(),
                    serde_json::to_value(&stage.metadata).map_err(|error| {
                        DagMlError::GraphValidation(format!(
                            "failed to serialize pipeline DSL generator `{}` stage `{}` metadata: {error}",
                            step.id, stage.id
                        ))
                    })?,
                );
            }
            options.push(GeneratedSequence {
                id: format!("{stage_index}:{}", branch.id),
                labels: vec![format!("{}:{}", stage.id, branch.id)],
                steps: branch.steps.clone(),
                metadata,
            });
        }
        stage_options.push(options);
    }

    let mut rows = Vec::<Vec<usize>>::new();
    build_cartesian_indices(&stage_options, 0, &mut Vec::new(), &mut rows, step.count);
    let mut choices = Vec::with_capacity(rows.len());
    for (choice_index, row) in rows.into_iter().enumerate() {
        let selected = row
            .into_iter()
            .enumerate()
            .map(|(stage_index, option_index)| stage_options[stage_index][option_index].clone())
            .collect::<Vec<_>>();
        choices.push(merge_generated_sequence(
            generator_choice_id(&step.id, choice_index),
            selected,
        )?);
    }
    Ok(choices)
}
pub(crate) fn generated_pick_sequences(
    options: &[GeneratedSequence],
    generator_id: &NodeId,
    mode: &str,
    sizes: &[usize],
    count: Option<usize>,
) -> Result<Vec<GeneratedSequence>> {
    let mut selections = Vec::<Vec<usize>>::new();
    for size in sizes {
        if *size == 0 || *size > options.len() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL generator `{generator_id}` {mode} size {size} is outside 1..={}",
                options.len()
            )));
        }
        build_combinations(
            options.len(),
            *size,
            0,
            &mut Vec::new(),
            &mut selections,
            count,
        );
    }
    selections
        .into_iter()
        .enumerate()
        .map(|(index, selection)| {
            let selected = selection
                .into_iter()
                .map(|option_index| options[option_index].clone())
                .collect::<Vec<_>>();
            merge_generated_sequence(generator_choice_id(generator_id, index), selected)
        })
        .collect()
}
pub(crate) fn generated_arrange_sequences(
    options: &[GeneratedSequence],
    generator_id: &NodeId,
    mode: &str,
    sizes: &[usize],
    count: Option<usize>,
) -> Result<Vec<GeneratedSequence>> {
    let mut selections = Vec::<Vec<usize>>::new();
    for size in sizes {
        if *size == 0 || *size > options.len() {
            return Err(DagMlError::GraphValidation(format!(
                "pipeline DSL generator `{generator_id}` {mode} size {size} is outside 1..={}",
                options.len()
            )));
        }
        build_permutations(
            options.len(),
            *size,
            &mut BTreeSet::new(),
            &mut Vec::new(),
            &mut selections,
            count,
        );
    }
    selections
        .into_iter()
        .enumerate()
        .map(|(index, selection)| {
            let selected = selection
                .into_iter()
                .map(|option_index| options[option_index].clone())
                .collect::<Vec<_>>();
            merge_generated_sequence(generator_choice_id(generator_id, index), selected)
        })
        .collect()
}
pub(crate) fn merge_generated_sequence(
    id: String,
    sequences: Vec<GeneratedSequence>,
) -> Result<GeneratedSequence> {
    if sequences.is_empty() {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generated sequence `{id}` has no selected options"
        )));
    }
    let mut labels = Vec::new();
    let mut steps = Vec::new();
    let mut metadata = BTreeMap::new();
    for sequence in sequences {
        labels.extend(sequence.labels);
        steps.extend(sequence.steps);
        if !sequence.metadata.is_empty() {
            metadata.insert(
                format!("option:{}", metadata.len()),
                serde_json::to_value(sequence.metadata).map_err(|error| {
                    DagMlError::GraphValidation(format!(
                        "failed to serialize generated sequence `{id}` metadata: {error}"
                    ))
                })?,
            );
        }
    }
    Ok(GeneratedSequence {
        id,
        labels,
        steps,
        metadata,
    })
}
pub(crate) fn truncate_generated_sequences(
    mut sequences: Vec<GeneratedSequence>,
    count: Option<usize>,
) -> Vec<GeneratedSequence> {
    if let Some(limit) = count {
        sequences.truncate(limit);
    }
    sequences
}
pub(crate) fn build_cartesian_indices<T>(
    stages: &[Vec<T>],
    stage_index: usize,
    current: &mut Vec<usize>,
    rows: &mut Vec<Vec<usize>>,
    count: Option<usize>,
) {
    if count.is_some_and(|limit| rows.len() >= limit) {
        return;
    }
    if stage_index == stages.len() {
        rows.push(current.clone());
        return;
    }
    for option_index in 0..stages[stage_index].len() {
        current.push(option_index);
        build_cartesian_indices(stages, stage_index + 1, current, rows, count);
        current.pop();
        if count.is_some_and(|limit| rows.len() >= limit) {
            break;
        }
    }
}
pub(crate) fn selection_sizes(
    selection: Option<PipelineDslSelectionSpec>,
) -> Result<Option<Vec<usize>>> {
    selection
        .map(|selection| match selection {
            PipelineDslSelectionSpec::Single(size) => {
                if size == 0 {
                    return Err(DagMlError::GraphValidation(
                        "pipeline DSL generator selection size cannot be zero".to_string(),
                    ));
                }
                Ok(vec![size])
            }
            PipelineDslSelectionSpec::Range([start, stop]) => {
                if start == 0 || stop == 0 || start > stop {
                    return Err(DagMlError::GraphValidation(format!(
                        "pipeline DSL generator selection range [{start}, {stop}] is invalid"
                    )));
                }
                Ok((start..=stop).collect())
            }
        })
        .transpose()
}
pub(crate) fn generator_choice_id(generator_id: &NodeId, choice_index: usize) -> String {
    format!("{generator_id}:choice{choice_index}")
}
pub(crate) fn generator_choice_metadata(
    step: &PipelineDslGeneratorStep,
    choice: &GeneratedSequence,
) -> Result<BTreeMap<String, serde_json::Value>> {
    let mut metadata = step.metadata.clone();
    metadata.insert(
        "dsl_generator".to_string(),
        serde_json::Value::String(step.id.to_string()),
    );
    metadata.insert(
        "dsl_generator_mode".to_string(),
        serde_json::to_value(step.mode).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL generator `{}` mode: {error}",
                step.id
            ))
        })?,
    );
    metadata.insert(
        "dsl_generator_choice".to_string(),
        serde_json::Value::String(choice.id.clone()),
    );
    metadata.insert(
        "dsl_generator_labels".to_string(),
        serde_json::to_value(&choice.labels).map_err(|error| {
            DagMlError::GraphValidation(format!(
                "failed to serialize pipeline DSL generator `{}` choice labels: {error}",
                step.id
            ))
        })?,
    );
    if !choice.metadata.is_empty() {
        metadata.insert(
            "dsl_generator_choice_metadata".to_string(),
            serde_json::to_value(&choice.metadata).map_err(|error| {
                DagMlError::GraphValidation(format!(
                    "failed to serialize pipeline DSL generator `{}` choice metadata: {error}",
                    step.id
                ))
            })?,
        );
    }
    Ok(metadata)
}
pub(crate) fn namespace_generated_sequence(
    generator: &PipelineDslGeneratorStep,
    mut choice: GeneratedSequence,
    choice_index: usize,
) -> Result<GeneratedSequence> {
    let mut node_map = BTreeMap::<NodeId, NodeId>::new();
    let mut counter = 0usize;
    for step in &mut choice.steps {
        namespace_step_ids(generator, choice_index, step, &mut counter, &mut node_map)?;
    }
    for step in &mut choice.steps {
        rewrite_step_node_refs(step, &node_map);
    }
    Ok(choice)
}
pub(crate) fn namespace_step_ids(
    generator: &PipelineDslGeneratorStep,
    choice_index: usize,
    step: &mut PipelineDslStep,
    counter: &mut usize,
    node_map: &mut BTreeMap<NodeId, NodeId>,
) -> Result<()> {
    match step {
        PipelineDslStep::Transform(step)
        | PipelineDslStep::YTransform(step)
        | PipelineDslStep::Tag(step)
        | PipelineDslStep::Exclude(step)
        | PipelineDslStep::Filter(step)
        | PipelineDslStep::SampleFilter(step)
        | PipelineDslStep::Augmentation(step)
        | PipelineDslStep::FeatureAugmentation(step)
        | PipelineDslStep::SampleAugmentation(step)
        | PipelineDslStep::DataGeneration(step)
        | PipelineDslStep::Model(step)
        | PipelineDslStep::Tuner(step)
        | PipelineDslStep::Chart(step) => {
            namespace_operator_step_id(generator, choice_index, step, counter, node_map)?;
        }
        PipelineDslStep::ConcatTransform(step) => {
            namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)?;
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    namespace_operator_step_id(
                        generator,
                        choice_index,
                        branch_step,
                        counter,
                        node_map,
                    )?;
                }
            }
        }
        PipelineDslStep::Branch(step) => {
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    namespace_step_ids(generator, choice_index, branch_step, counter, node_map)?;
                }
            }
        }
        PipelineDslStep::Generator(step) => {
            namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)?;
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    namespace_step_ids(generator, choice_index, branch_step, counter, node_map)?;
                }
            }
            for stage in &mut step.stages {
                for branch in &mut stage.branches {
                    for branch_step in &mut branch.steps {
                        namespace_step_ids(
                            generator,
                            choice_index,
                            branch_step,
                            counter,
                            node_map,
                        )?;
                    }
                }
            }
        }
        PipelineDslStep::Sequential(step) => {
            if let Some(id) = &mut step.id {
                namespace_node_id_field(generator, choice_index, id, counter, node_map)?;
            }
            for child in &mut step.steps {
                namespace_step_ids(generator, choice_index, child, counter, node_map)?;
            }
        }
        PipelineDslStep::Merge(step) => {
            namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)?;
        }
        PipelineDslStep::MergeModel(step) => {
            namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)?;
        }
    }
    Ok(())
}
pub(crate) fn namespace_operator_step_id(
    generator: &PipelineDslGeneratorStep,
    choice_index: usize,
    step: &mut PipelineDslOperatorStep,
    counter: &mut usize,
    node_map: &mut BTreeMap<NodeId, NodeId>,
) -> Result<()> {
    namespace_node_id_field(generator, choice_index, &mut step.id, counter, node_map)
}
pub(crate) fn namespace_node_id_field(
    generator: &PipelineDslGeneratorStep,
    choice_index: usize,
    node_id: &mut NodeId,
    counter: &mut usize,
    node_map: &mut BTreeMap<NodeId, NodeId>,
) -> Result<()> {
    let original = node_id.clone();
    if node_map.contains_key(&original) {
        return Err(DagMlError::GraphValidation(format!(
            "pipeline DSL generator `{}` choice `{}` reuses node id `{original}`; generated choices require unique node ids inside each expanded sequence",
            generator.id, choice_index
        )));
    }
    let next = namespaced_generated_node_id(&generator.id, choice_index, *counter, &original)?;
    *counter += 1;
    *node_id = next.clone();
    node_map.insert(original, next);
    Ok(())
}
pub(crate) fn namespaced_generated_node_id(
    generator_id: &NodeId,
    choice_index: usize,
    node_index: usize,
    original: &NodeId,
) -> Result<NodeId> {
    let generator = sanitized_id_fragment(generator_id.as_str(), 32);
    let suffix = sanitized_id_fragment(original.as_str(), 28);
    NodeId::new(format!(
        "gen:{generator}:c{choice_index}:n{node_index}.{suffix}"
    ))
}
pub(crate) fn sanitized_id_fragment(input: &str, max_len: usize) -> String {
    let sanitized = sanitize_generation_label(input);
    let mut fragment = sanitized.chars().take(max_len).collect::<String>();
    if fragment.is_empty() {
        fragment = "x".to_string();
    }
    fragment
}
pub(crate) fn rewrite_step_node_refs(
    step: &mut PipelineDslStep,
    node_map: &BTreeMap<NodeId, NodeId>,
) {
    match step {
        PipelineDslStep::Transform(_)
        | PipelineDslStep::YTransform(_)
        | PipelineDslStep::Tag(_)
        | PipelineDslStep::Exclude(_)
        | PipelineDslStep::Filter(_)
        | PipelineDslStep::SampleFilter(_)
        | PipelineDslStep::Augmentation(_)
        | PipelineDslStep::FeatureAugmentation(_)
        | PipelineDslStep::SampleAugmentation(_)
        | PipelineDslStep::DataGeneration(_)
        | PipelineDslStep::Model(_)
        | PipelineDslStep::Tuner(_)
        | PipelineDslStep::Chart(_) => {}
        PipelineDslStep::ConcatTransform(step) => {
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    rewrite_operator_step_refs(branch_step, node_map);
                }
            }
        }
        PipelineDslStep::Branch(step) => {
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    rewrite_step_node_refs(branch_step, node_map);
                }
            }
        }
        PipelineDslStep::Generator(step) => {
            for branch in &mut step.branches {
                for branch_step in &mut branch.steps {
                    rewrite_step_node_refs(branch_step, node_map);
                }
            }
            for stage in &mut step.stages {
                for branch in &mut stage.branches {
                    for branch_step in &mut branch.steps {
                        rewrite_step_node_refs(branch_step, node_map);
                    }
                }
            }
        }
        PipelineDslStep::Sequential(step) => {
            for child in &mut step.steps {
                rewrite_step_node_refs(child, node_map);
            }
        }
        PipelineDslStep::Merge(step) => {
            rewrite_merge_selectors(&mut step.selectors, node_map);
        }
        PipelineDslStep::MergeModel(_) => {}
    }
}
pub(crate) fn rewrite_operator_step_refs(
    _step: &mut PipelineDslOperatorStep,
    _node_map: &BTreeMap<NodeId, NodeId>,
) {
}
