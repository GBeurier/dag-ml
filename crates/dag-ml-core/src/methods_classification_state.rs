//! Independent bounded wire validation. Numerical hydration remains Methods-owned.
use std::collections::BTreeMap;

use serde_json::Value;

use crate::{DagMlError, Result};

fn invalid<T>() -> Result<T> {
    Err(DagMlError::RuntimeValidation("classifier learned state differs from its signed recipe, class order or bounded wire format".into()))
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}
impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| DagMlError::RuntimeValidation("truncated classifier state".into()))?;
        let value = &self.bytes[self.position..end];
        self.position = end;
        Ok(value)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }
    fn f64(&mut self) -> Result<f64> {
        let value = f64::from_bits(self.u64()?);
        if !value.is_finite() {
            return invalid();
        }
        Ok(value)
    }
    fn block(&mut self) -> Result<&'a [u8]> {
        let count = self.u64()?;
        if count > 67_108_864 {
            return invalid();
        }
        self.take(count as usize)
    }
    fn string(&mut self) -> Result<String> {
        let bytes = self.block()?;
        if bytes.len() > 1_048_576 {
            return invalid();
        }
        String::from_utf8(bytes.to_vec())
            .map_err(|_| DagMlError::RuntimeValidation("classifier state text is not UTF-8".into()))
    }
    fn short_string(&mut self) -> Result<String> {
        let count = self.u32()?;
        if count > 4096 {
            return invalid();
        }
        String::from_utf8(self.take(count as usize)?.to_vec())
            .map_err(|_| DagMlError::RuntimeValidation("estimator state text is not UTF-8".into()))
    }
    fn end(&self) -> Result<()> {
        if self.position == self.bytes.len() {
            Ok(())
        } else {
            invalid()
        }
    }
}

fn checksum<'a>(bytes: &'a [u8], magic: &[u8]) -> Result<&'a [u8]> {
    if bytes.len() < 28 || bytes.len() > 67_108_864 || !bytes.starts_with(magic) {
        return invalid();
    }
    let body = &bytes[..bytes.len() - 8];
    let hash = body.iter().fold(0xcbf29ce484222325_u64, |value, byte| {
        (value ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    });
    if hash != u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().expect("checksum bytes")) {
        return invalid();
    }
    Ok(body)
}

/// Writer ABI is distinct from the wire version. This standalone validator has
/// no loaded runtime: admission remains caller/Methods-owned, including N4MC's
/// upper minor bound on hydration. Patch does not select a wire layout.
fn writer_abi(input: &mut Reader<'_>, minimum_minor: u32) -> Result<(u32, u32)> {
    let major = input.u32()?;
    let minor = input.u32()?;
    let _patch = input.u32()?;
    if major != crate::hpo::METHODS_ABI_MAJOR || minor < minimum_minor {
        return invalid();
    }
    Ok((major, minor))
}

/// Project runtime requirements from an already owner-validated N4MC state.
/// This header projection creates no scientific proof and does not repeat body
/// or checksum validation. Unlike N4ME/N4MM, Methods classifier_load requires
/// runtime minor >= writer minor, so the producer minor is an admission minimum.
pub(crate) fn validated_n4mc_abi_requirement(bytes: &[u8]) -> Result<(u32, u32)> {
    let mut input = Reader::new(bytes);
    if input.take(4)? != b"N4MC" || input.u32()? != 1 {
        return invalid();
    }
    writer_abi(&mut input, 17)
}

struct EstimatorState<'a> {
    writer_minor: u32,
    params: BTreeMap<String, (u32, u64)>,
    capabilities: u64,
    width: u64,
    outputs: u64,
    blocks: BTreeMap<u32, &'a [u8]>,
}
fn estimator<'a>(bytes: &'a [u8], method: &str) -> Result<EstimatorState<'a>> {
    let mut input = Reader::new(checksum(bytes, b"N4ME")?);
    input.take(4)?;
    if input.u32()? != 1 {
        return invalid();
    }
    let minimum_minor = if method == crate::METHODS_CLASSIFIER_METHOD {
        17 // The DAG role-classifier profile requires Methods ABI 2.17.
    } else {
        crate::hpo::METHODS_N4ME_MIN_ABI_MINOR
    };
    let (_, writer_minor) = writer_abi(&mut input, minimum_minor)?;
    if input.short_string()? != method {
        return invalid();
    }
    let count = input.u32()?;
    if count > 256 {
        return invalid();
    }
    let mut params = BTreeMap::new();
    for _ in 0..count {
        let name = input.short_string()?;
        let kind = input.u32()?;
        if input.u64()? != 1
            || !matches!(kind, 1..=4)
            || params.insert(name, (kind, input.u64()?)).is_some()
        {
            return invalid();
        }
    }
    let capabilities = input.u64()?;
    let width = input.u64()?;
    let outputs = input.u64()?;
    if width == 0 || width > 1_048_576 || outputs > 1_048_576 || capabilities & 512 != 0 {
        return invalid();
    }
    let count = input.u32()?;
    if count == 0 || count > 256 {
        return invalid();
    }
    let mut blocks = BTreeMap::new();
    for _ in 0..count {
        let tag = input.u32()?;
        if blocks.insert(tag, input.block()?).is_some() {
            return invalid();
        }
    }
    input.end()?;
    Ok(EstimatorState {
        writer_minor,
        params,
        capabilities,
        width,
        outputs,
        blocks,
    })
}

fn validate_latent_model(bytes: &[u8], width: u64, components: u64, classes: u64) -> Result<()> {
    let mut input = Reader::new(checksum(bytes, b"N4MM")?);
    input.take(4)?;
    let format = input.u32()?;
    let minimum_minor = match format {
        1 => crate::hpo::METHODS_PLS_N4MM_MIN_ABI_MINOR,
        3 => crate::hpo::METHODS_COMPONENT_RESOLUTION_N4MM_MIN_ABI_MINOR,
        // Format 2 embeds preprocessing, absent from this latent-head recipe.
        _ => return invalid(),
    };
    writer_abi(&mut input, minimum_minor)?;
    if [input.u32()?, input.u32()?, input.u32()?] != [0, 1, 0] {
        return invalid();
    }
    let samples = input.u64()?;
    if samples <= components
        || input.u64()? != width
        || input.u64()? != classes
        || input.u64()? != components
        || [
            input.u32()?,
            input.u32()?,
            input.u32()?,
            input.u32()?,
            input.u32()?,
        ] != [1, 0, 1, 0, 0]
        || input.f64()? <= 0.0
        || input.u32()? == 0
    {
        return invalid();
    }
    let product = |left: u64, right: u64| {
        left.checked_mul(right)
            .ok_or_else(|| DagMlError::RuntimeValidation("latent model size overflow".into()))
    };
    for count in [
        width,
        width,
        classes,
        classes,
        product(width, classes)?,
        product(width, components)?,
        product(width, components)?,
        product(classes, components)?,
        product(width, components)?,
        0,
        0,
    ] {
        if input.u64()? != count {
            return invalid();
        }
        for _ in 0..count {
            input.f64()?;
        }
    }
    if format == 3 {
        // Match Methods read_resolution_v3: one finite nonnegative resolution,
        // one +0/1 unresolved flag per component, then pipeline presence.
        if input.u64()? != 1 {
            return invalid();
        }
        let resolution = input.f64()?;
        if resolution < 0.0 || resolution.is_sign_negative() || input.u64()? != components {
            return invalid();
        }
        for _ in 0..components {
            let flag = input.f64()?;
            if !(flag == 1.0 || (flag == 0.0 && !flag.is_sign_negative())) {
                return invalid();
            }
        }
        if input.u32()? != 0 {
            return invalid(); // No pipeline is part of this classifier latent model.
        }
    }
    input.end()
}

pub(crate) fn validate_classifier_n4me(
    bytes: &[u8],
    params: &Value,
    width: usize,
    classes: &[usize],
) -> Result<()> {
    let state = estimator(bytes, crate::METHODS_CLASSIFIER_METHOD)?;
    let components = params["n_components"]
        .as_u64()
        .ok_or_else(|| DagMlError::RuntimeValidation("classifier components missing".into()))?;
    let iterations = params["max_iter"]
        .as_u64()
        .ok_or_else(|| DagMlError::RuntimeValidation("classifier iterations missing".into()))?;
    if state.width != width as u64
        || state.outputs != classes.len() as u64
        || state.capabilities != 156
        || state.params
            != BTreeMap::from([
                ("n_components".into(), (1, components)),
                ("max_iter".into(), (1, iterations)),
            ])
        || state.blocks.len() != 1
    {
        return invalid();
    }
    let mut head =
        Reader::new(state.blocks.get(&0x31534c43).ok_or_else(|| {
            DagMlError::RuntimeValidation("classifier CLS1 block missing".into())
        })?);
    if head.i64()? != width as i64 || head.u64()? != classes.len() as u64 {
        return invalid();
    }
    for id in classes {
        if head.i64()? != *id as i64 {
            return invalid();
        }
    }
    let model_size = head.i64()?;
    if !(28..=67_108_864).contains(&model_size) {
        return invalid();
    }
    let packed_count = head.u64()?;
    if packed_count != (model_size as u64).div_ceil(8) {
        return invalid();
    }
    let packed = head.take(packed_count as usize * 8)?;
    if !packed.starts_with(b"N4MM") || packed[model_size as usize..].iter().any(|byte| *byte != 0) {
        return invalid();
    }
    validate_latent_model(
        &packed[..model_size as usize],
        width as u64,
        components,
        classes.len() as u64,
    )?;
    let tail = classes.len() as u64 - 1;
    for count in [
        tail,
        tail.checked_mul(components).ok_or_else(|| {
            DagMlError::RuntimeValidation("classifier coefficient width overflow".into())
        })?,
    ] {
        if head.u64()? != count {
            return invalid();
        }
        for _ in 0..count {
            head.f64()?;
        }
    }
    head.end()
}

/// First Methods ABI minor whose `standard_scale` resolves the integer `ddof`
/// (nirs4all-methods 1.3.15). It landed inside the 2.22 cycle without an ABI
/// bump, so only 2.22 writers may emit either parameter set.
const METHODS_STANDARD_SCALE_DDOF_ABI_MINOR: u32 = 22;

/// The closed `standard_scaler` and `column_transformer` recipes declare no
/// `ddof`: they sign population scaling, the only convention before the
/// parameter existed and the one Methods requires of every nested scaler.
const RECIPE_STANDARD_SCALE_DDOF: u64 = 0;

/// `ddof` is fit provenance: the divisor `rows - ddof` behind the stored scales.
/// It cannot be recomputed from the state, so the record must agree exactly.
fn validate_standard_scale_ddof(state: &mut EstimatorState<'_>) -> Result<()> {
    let minor = state.writer_minor;
    let refuse = |detail: String| {
        Err(DagMlError::RuntimeValidation(format!(
            "classifier standard_scale state {detail}"
        )))
    };
    match state.params.remove("ddof") {
        None if minor > METHODS_STANDARD_SCALE_DDOF_ABI_MINOR => refuse(format!(
            "omits ddof, which every Methods ABI 2.{minor} writer records"
        )),
        None => Ok(()),
        Some(_) if minor < METHODS_STANDARD_SCALE_DDOF_ABI_MINOR => refuse(format!(
            "records ddof under Methods ABI 2.{minor}, which predates the parameter"
        )),
        Some((1, RECIPE_STANDARD_SCALE_DDOF)) => Ok(()),
        Some((1, ddof)) => refuse(format!(
            "records ddof={}, but its signed recipe implies population scaling (ddof={RECIPE_STANDARD_SCALE_DDOF})",
            ddof as i64
        )),
        Some(_) => invalid(),
    }
}

fn validate_encoder(bytes: &[u8], encoder: &Value, input_width: u64) -> Result<u64> {
    let pca = encoder["kind"] == "tensor_pca";
    let method = if pca {
        "preprocessing.feature_selection.flexible_pca"
    } else {
        "preprocessing.scaling.standard_scale"
    };
    let mut state = estimator(bytes, method)?;
    if !pca {
        validate_standard_scale_ddof(&mut state)?;
    }
    let expected = if pca {
        BTreeMap::from([(
            "n_components".into(),
            (
                2,
                (encoder["n_components"].as_u64().unwrap_or(0) as f64).to_bits(),
            ),
        )])
    } else {
        BTreeMap::from([("with_mean".into(), (3, 1)), ("with_std".into(), (3, 1))])
    };
    if state.width != input_width
        || state.outputs != 0
        || state.capabilities != 129
        || state.params != expected
        || state.blocks.len() != 1
    {
        return invalid();
    }
    let output = if pca {
        encoder["n_components"].as_u64().unwrap_or(0)
    } else {
        input_width
    };
    let mut learned = Reader::new(state.blocks.get(&0x31545354).ok_or_else(|| {
        DagMlError::RuntimeValidation("classifier encoder TST1 block missing".into())
    })?);
    if learned.i64()? != input_width as i64 {
        return invalid();
    }
    if pca {
        if learned.u64()? != input_width {
            return invalid();
        }
        for _ in 0..input_width {
            learned.f64()?;
        }
        if learned.i64()? != output as i64 || learned.u64()? != input_width * output {
            return invalid();
        }
        for _ in 0..input_width * output {
            learned.f64()?;
        }
    } else {
        if learned.i64()? != 1 || learned.i64()? != 1 {
            return invalid();
        }
        for positive in [false, true] {
            if learned.u64()? != input_width {
                return invalid();
            }
            for _ in 0..input_width {
                if learned.f64()? <= 0.0 && positive {
                    return invalid();
                }
            }
        }
    }
    learned.end()?;
    Ok(output)
}

fn put_string(output: &mut Vec<u8>, text: &str) {
    output.extend((text.len() as u64).to_le_bytes());
    output.extend(text.as_bytes());
}

fn canonical_recipe(recipe: &Value, schemas: &Value) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    put_string(&mut output, crate::METHODS_CLASSIFIER_METHOD);
    for name in ["n_components", "max_iter"] {
        output.extend(
            recipe["model"]["params"][name]
                .as_u64()
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation("classifier parameters missing".into())
                })?
                .to_le_bytes(),
        );
    }
    let order = recipe["source_order"]
        .as_array()
        .ok_or_else(|| DagMlError::RuntimeValidation("classifier source order missing".into()))?;
    output.extend((order.len() as u32).to_le_bytes());
    for value in order {
        let name = value.as_str().ok_or_else(|| {
            DagMlError::RuntimeValidation("classifier source name missing".into())
        })?;
        let schema = &schemas[name];
        let encoder = &recipe["encoders"][name];
        for text in [
            name,
            schema["representation_id"].as_str().unwrap_or(""),
            schema["dtype"].as_str().unwrap_or(""),
            schema["identity"].as_str().unwrap_or(""),
        ] {
            put_string(&mut output, text);
        }
        let shape = schema["input_shape"].as_array().ok_or_else(|| {
            DagMlError::RuntimeValidation("classifier source shape missing".into())
        })?;
        output.extend((shape.len() as u32).to_le_bytes());
        for dimension in shape {
            output.extend(dimension.as_u64().unwrap_or(0).to_le_bytes());
        }
        let kind = match encoder["kind"].as_str() {
            Some("standard_scaler") => 1_u32,
            Some("tensor_pca") => 2,
            Some("column_transformer") => 3,
            _ => return invalid(),
        };
        output.extend(kind.to_le_bytes());
        output.extend(
            recipe["source_weights"][name]
                .as_f64()
                .ok_or_else(|| DagMlError::RuntimeValidation("classifier weight missing".into()))?
                .to_le_bytes(),
        );
        for value in [
            encoder["n_components"].as_u64().unwrap_or(0),
            encoder["random_state"].as_u64().unwrap_or(0),
            if kind == 3 { 0 } else { u64::MAX },
            if kind == 3 { 1 } else { u64::MAX },
        ] {
            output.extend(value.to_le_bytes());
        }
        for value in [
            u32::from(kind != 2),
            u32::from(kind != 2),
            0,
            u32::from(kind == 3),
        ] {
            output.extend(value.to_le_bytes());
        }
    }
    Ok(output)
}

pub(crate) fn validate_classifier_n4mc(
    bytes: &[u8],
    recipe: &Value,
    schemas: &Value,
    classes: &[usize],
) -> Result<()> {
    let mut input = Reader::new(checksum(bytes, b"N4MC")?);
    validated_n4mc_abi_requirement(input.take(20)?)?;
    if input.block()? != canonical_recipe(recipe, schemas)? {
        return invalid();
    }
    let mut width = 0_u64;
    for source in recipe["source_order"]
        .as_array()
        .ok_or_else(|| DagMlError::RuntimeValidation("classifier source order missing".into()))?
    {
        let name = source.as_str().ok_or_else(|| {
            DagMlError::RuntimeValidation("classifier source name missing".into())
        })?;
        let encoder = &recipe["encoders"][name];
        let features = if name == "metadata" {
            1
        } else {
            schemas[name]["input_shape"]
                .as_array()
                .ok_or_else(|| {
                    DagMlError::RuntimeValidation("classifier source shape missing".into())
                })?
                .iter()
                .try_fold(1_u64, |count, size| {
                    count
                        .checked_mul(size.as_u64().unwrap_or(0))
                        .ok_or_else(|| {
                            DagMlError::RuntimeValidation("classifier input width overflow".into())
                        })
                })?
        };
        let encoded = validate_encoder(input.block()?, encoder, features)?;
        let categories = input.u64()?;
        if categories > 65_536
            || (name == "metadata" && categories == 0)
            || (name != "metadata" && categories != 0)
        {
            return invalid();
        }
        let mut previous: Option<String> = None;
        for _ in 0..categories {
            let value = input.string()?;
            if previous.as_ref().is_some_and(|prior| prior >= &value) {
                return invalid();
            }
            previous = Some(value);
        }
        width = width.checked_add(encoded + categories).ok_or_else(|| {
            DagMlError::RuntimeValidation("classifier encoded width overflow".into())
        })?;
    }
    if input.u64()? != classes.len() as u64 {
        return invalid();
    }
    for class in classes {
        if input.i64()? != *class as i64 {
            return invalid();
        }
    }
    validate_classifier_n4me(
        input.block()?,
        &recipe["model"]["params"],
        width as usize,
        classes,
    )?;
    input.end()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Structural wire witnesses only, not fitted numerical state qualification.
    fn seal(mut bytes: Vec<u8>) -> Vec<u8> {
        let hash = bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        });
        bytes.extend(hash.to_le_bytes());
        bytes
    }

    fn header(magic: &[u8], format: u32, minor: u32) -> Vec<u8> {
        let mut bytes = magic.to_vec();
        for field in [format, 2, minor, 0] {
            bytes.extend(field.to_le_bytes());
        }
        bytes
    }

    fn block(bytes: &mut Vec<u8>, payload: &[u8]) {
        bytes.extend((payload.len() as u64).to_le_bytes());
        bytes.extend(payload);
    }

    fn short_string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend((text.len() as u32).to_le_bytes());
        bytes.extend(text.as_bytes());
    }

    fn latent(format: u32, minor: u32) -> Vec<u8> {
        let mut bytes = header(b"N4MM", format, minor);
        for field in [0_u32, 1, 0] {
            bytes.extend(field.to_le_bytes());
        }
        for field in [4_u64, 2, 2, 1] {
            bytes.extend(field.to_le_bytes());
        }
        for field in [1_u32, 0, 1, 0, 0] {
            bytes.extend(field.to_le_bytes());
        }
        bytes.extend(1e-8_f64.to_le_bytes());
        bytes.extend(1000_u32.to_le_bytes());
        for count in [2_u64, 2, 2, 2, 4, 2, 2, 2, 2, 0, 0] {
            bytes.extend(count.to_le_bytes());
            for _ in 0..count {
                bytes.extend(0.0_f64.to_le_bytes());
            }
        }
        if format == 3 {
            bytes.extend(1_u64.to_le_bytes());
            bytes.extend(0.0_f64.to_le_bytes());
            bytes.extend(1_u64.to_le_bytes());
            bytes.extend(0.0_f64.to_le_bytes());
            bytes.extend(0_u32.to_le_bytes());
        }
        seal(bytes)
    }

    fn estimator_state(classifier: bool, minor: u32, model: &[u8]) -> Vec<u8> {
        let params: &[(&str, u32, u64)] = if classifier {
            &[("n_components", 1, 1), ("max_iter", 1, 1000)]
        } else {
            &[("with_mean", 3, 1), ("with_std", 3, 1)]
        };
        estimator_state_with(classifier, minor, model, params)
    }

    fn scaler_state(minor: u32, ddof: Option<(u32, u64)>) -> Vec<u8> {
        let mut params = vec![("with_mean", 3_u32, 1_u64), ("with_std", 3, 1)];
        params.extend(ddof.map(|(kind, value)| ("ddof", kind, value)));
        estimator_state_with(false, minor, &[], &params)
    }

    fn estimator_state_with(
        classifier: bool,
        minor: u32,
        model: &[u8],
        params: &[(&str, u32, u64)],
    ) -> Vec<u8> {
        let mut bytes = header(b"N4ME", 1, minor);
        short_string(
            &mut bytes,
            if classifier {
                crate::METHODS_CLASSIFIER_METHOD
            } else {
                "preprocessing.scaling.standard_scale"
            },
        );
        bytes.extend((params.len() as u32).to_le_bytes());
        for &(name, kind, value) in params {
            short_string(&mut bytes, name);
            bytes.extend(kind.to_le_bytes());
            bytes.extend(1_u64.to_le_bytes());
            bytes.extend(value.to_le_bytes());
        }
        for value in [
            if classifier { 156_u64 } else { 129 },
            2,
            if classifier { 2 } else { 0 },
        ] {
            bytes.extend(value.to_le_bytes());
        }
        let mut learned = Vec::new();
        if classifier {
            for value in [
                2_u64,
                2,
                0,
                1,
                model.len() as u64,
                (model.len() as u64).div_ceil(8),
            ] {
                learned.extend(value.to_le_bytes());
            }
            learned.extend(model);
            learned.resize(learned.len().next_multiple_of(8), 0);
            for _ in 0..2 {
                learned.extend(1_u64.to_le_bytes());
                learned.extend(0.0_f64.to_le_bytes());
            }
        } else {
            for value in [2_i64, 1, 1] {
                learned.extend(value.to_le_bytes());
            }
            for value in [0.0_f64, 1.0] {
                learned.extend(2_u64.to_le_bytes());
                learned.extend(value.to_le_bytes());
                learned.extend(value.to_le_bytes());
            }
        }
        bytes.extend(1_u32.to_le_bytes());
        bytes.extend(
            (if classifier {
                0x31534c43_u32
            } else {
                0x31545354
            })
            .to_le_bytes(),
        );
        block(&mut bytes, &learned);
        seal(bytes)
    }

    fn classifier_state(format: u32, minor: u32) -> (Vec<u8>, Value, Value) {
        let recipe = json!({"schema_version":1,"fusion":"early","source_order":["nir"],
            "encoders":{"nir":{"kind":"standard_scaler","with_mean":true,"with_std":true}},
            "source_weights":{"nir":1.0},"model":{"method_id":crate::METHODS_CLASSIFIER_METHOD,
            "params":{"n_components":1,"max_iter":1000}}});
        let schemas = json!({"nir":{"representation_id":"signal_1d","input_shape":[2],
            "dtype":"float64","identity":"{}"}});
        let mut bytes = header(b"N4MC", 1, minor);
        block(&mut bytes, &canonical_recipe(&recipe, &schemas).unwrap());
        block(&mut bytes, &estimator_state(false, minor, &[]));
        for value in [0_u64, 2, 0, 1] {
            bytes.extend(value.to_le_bytes());
        }
        block(
            &mut bytes,
            &estimator_state(true, minor, &latent(format, minor)),
        );
        (seal(bytes), recipe, schemas)
    }

    fn change_header(mut bytes: Vec<u8>, offset: usize, value: u32) -> Vec<u8> {
        bytes.truncate(bytes.len() - 8);
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        seal(bytes)
    }

    #[test]
    fn classifier_known_wires_accept_abi_220_and_reject_incompatible_or_tampered_state() {
        for (minor, format) in [(17, 1), (20, 1), (20, 3), (21, 3)] {
            // A known wire from a future writer can be structurally valid.
            // This is not admission: Methods N4MC hydration checks runtime minor.
            let (bytes, recipe, schemas) = classifier_state(format, minor);
            validate_classifier_n4mc(&bytes, &recipe, &schemas, &[0, 1]).unwrap();
            let (major, required_minor) = validated_n4mc_abi_requirement(&bytes).unwrap();
            assert_eq!((major, required_minor), (2, minor));
            // Same comparison consumed by Core before native import; no runtime
            // is loaded here and no future runtime version is assumed.
            let admitted =
                crate::hpo::validate_methods_abi_compatibility(2, 20, major, required_minor);
            assert_eq!(admitted.is_ok(), minor <= 20);
        }
        let (bytes, recipe, schemas) = classifier_state(3, 20);
        validate_classifier_n4mc(
            &change_header(bytes.clone(), 16, 9),
            &recipe,
            &schemas,
            &[0, 1],
        )
        .unwrap();
        let encoder = &recipe["encoders"]["nir"];
        validate_encoder(&estimator_state(false, 13, &[]), encoder, 2).unwrap();
        assert!(validate_encoder(&estimator_state(false, 12, &[]), encoder, 2).is_err());
        for (offset, value) in [(4, 2), (8, 3), (12, 16)] {
            let changed = change_header(bytes.clone(), offset, value);
            assert!(validate_classifier_n4mc(&changed, &recipe, &schemas, &[0, 1]).is_err());
            assert!(validated_n4mc_abi_requirement(&changed).is_err());
        }
        assert!(validated_n4mc_abi_requirement(&bytes[..19]).is_err());
        let mut corrupt = bytes.clone();
        corrupt[20] ^= 1;
        assert!(validate_classifier_n4mc(&corrupt, &recipe, &schemas, &[0, 1]).is_err());
        let mut changed_recipe = recipe.clone();
        changed_recipe["source_weights"]["nir"] = json!(2.0);
        assert!(validate_classifier_n4mc(&bytes, &changed_recipe, &schemas, &[0, 1]).is_err());
        assert!(validate_classifier_n4mc(&bytes, &recipe, &schemas, &[1, 0]).is_err());

        let model = latent(3, 20);
        for (offset, value) in [(4, 2), (4, 4), (8, 3), (12, 19)] {
            assert!(
                validate_latent_model(&change_header(model.clone(), offset, value), 2, 1, 2)
                    .is_err()
            );
        }
        let mut malformed = model[..model.len() - 8].to_vec();
        let flag_offset = malformed.len() - 12;
        malformed[flag_offset..flag_offset + 8].copy_from_slice(&(-0.0_f64).to_le_bytes());
        assert!(validate_latent_model(&seal(malformed), 2, 1, 2).is_err());
        let pipeline_offset = model.len() - 12;
        assert!(
            validate_latent_model(&change_header(model.clone(), pipeline_offset, 1), 2, 1, 2)
                .is_err()
        );
        let estimator = estimator_state(true, 20, &model);
        for (offset, value) in [(4, 2), (8, 3), (12, 16)] {
            assert!(validate_classifier_n4me(
                &change_header(estimator.clone(), offset, value),
                &recipe["model"]["params"],
                2,
                &[0, 1]
            )
            .is_err());
        }
    }

    /// Genuine `MultimodalClassifierPipeline.export_state()` bytes; see the
    /// fixture README for provenance.
    fn methods_state(fixture: &str) -> (Vec<u8>, Value, Value, Vec<usize>) {
        let fixture: Value = serde_json::from_str(fixture).unwrap();
        let hex = fixture["state_hex"].as_str().unwrap();
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).unwrap())
            .collect();
        let classes = serde_json::from_value(fixture["class_labels"].clone()).unwrap();
        (
            bytes,
            fixture["recipe"].clone(),
            fixture["source_schemas"].clone(),
            classes,
        )
    }

    /// Rewrite the `ddof` value of the first nested scaler and restore both
    /// checksums, as a writer fitted with another convention would emit it.
    fn with_first_scaler_ddof(state: &[u8], ddof: u64) -> Vec<u8> {
        let recipe = 20 + 8 + u64::from_le_bytes(state[20..28].try_into().unwrap()) as usize;
        let size = u64::from_le_bytes(state[recipe..recipe + 8].try_into().unwrap()) as usize;
        let (start, end) = (recipe + 8, recipe + 8 + size);
        let mut encoder = state[start..end - 8].to_vec();
        let name = encoder
            .windows(4)
            .position(|bytes| bytes == b"ddof")
            .unwrap();
        encoder[name + 16..name + 24].copy_from_slice(&ddof.to_le_bytes());
        let mut changed = state[..start].to_vec();
        changed.extend(seal(encoder));
        changed.extend(&state[end..state.len() - 8]);
        seal(changed)
    }

    #[test]
    fn genuine_methods_states_validate_before_and_after_standard_scale_ddof() {
        for (fixture, minor) in [
            (
                include_str!("../tests/fixtures/methods_classifier_state/methods_1_3_14.json"),
                20,
            ),
            (
                include_str!("../tests/fixtures/methods_classifier_state/methods_1_3_15.json"),
                22,
            ),
            (
                include_str!("../tests/fixtures/methods_classifier_state/methods_1_3_18.json"),
                23,
            ),
        ] {
            let (bytes, recipe, schemas, classes) = methods_state(fixture);
            assert_eq!(validated_n4mc_abi_requirement(&bytes).unwrap(), (2, minor));
            validate_classifier_n4mc(&bytes, &recipe, &schemas, &classes).unwrap();
        }
    }

    #[test]
    fn genuine_methods_state_fitted_with_another_ddof_is_refused() {
        let (bytes, recipe, schemas, classes) = methods_state(include_str!(
            "../tests/fixtures/methods_classifier_state/methods_1_3_18.json"
        ));
        let unchanged = with_first_scaler_ddof(&bytes, 0);
        assert_eq!(unchanged, bytes);
        let sample = with_first_scaler_ddof(&bytes, 1);
        let error = validate_classifier_n4mc(&sample, &recipe, &schemas, &classes).unwrap_err();
        assert!(
            error.to_string().contains(
                "records ddof=1, but its signed recipe implies population scaling (ddof=0)"
            ),
            "{error}"
        );
    }

    #[test]
    fn standard_scale_ddof_is_exact_for_each_writer_generation() {
        let encoder = json!({"kind":"standard_scaler","with_mean":true,"with_std":true});
        let check = |minor, ddof| validate_encoder(&scaler_state(minor, ddof), &encoder, 2);
        // Before the parameter existed, population scaling was the only convention.
        for minor in [13, 20, 21, 22] {
            assert_eq!(check(minor, None).unwrap(), 2);
        }
        // The parameter landed inside the ABI 2.22 cycle without a bump.
        for minor in [22, 23, 40] {
            assert_eq!(check(minor, Some((1, 0))).unwrap(), 2);
        }
        for (minor, ddof, message) in [
            (
                22,
                Some((1, 1)),
                "records ddof=1, but its signed recipe implies population",
            ),
            (
                23,
                Some((1, u64::MAX)),
                "records ddof=-1, but its signed recipe implies population",
            ),
            (
                23,
                None,
                "omits ddof, which every Methods ABI 2.23 writer records",
            ),
            (
                21,
                Some((1, 0)),
                "records ddof under Methods ABI 2.21, which predates the parameter",
            ),
        ] {
            let error = check(minor, ddof).unwrap_err().to_string();
            assert!(error.contains(message), "{error}");
        }
        // `ddof` is an integer parameter: no other kind can restate zero.
        for kind in [2, 3, 4] {
            assert!(check(23, Some((kind, 0))).is_err());
        }
        // The metadata numeric column uses the same nested scaler.
        let metadata = json!({"kind":"column_transformer","numeric_columns":[0],
            "categorical_columns":[1],"with_mean":true,"with_std":true,
            "handle_unknown":"ignore","sparse_output":false,"drop":null});
        assert!(validate_encoder(&scaler_state(23, Some((1, 2))), &metadata, 2).is_err());
        validate_encoder(&scaler_state(23, Some((1, 0))), &metadata, 2).unwrap();
    }
}
