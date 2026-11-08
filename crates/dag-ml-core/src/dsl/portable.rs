//! Admission for the declared linear portable profile. No buffers or operators run here.
use super::*;
use serde_json::{json, Map, Value};

pub const PORTABLE_LINEAR_PROFILE: &str = "portable_linear_regression.v1";
pub const PORTABLE_STEPS_CONTROLLER: &str = "controller:methods.native.pipeline_steps";
pub const PORTABLE_STEPS_VERSION: &str = "1.1.0";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PortableLinearAdmission {
    pub schema_version: u32,
    pub profile: String,
    pub owner_profile: String,
    pub mode: String,
    pub name: String,
    pub splitter: Option<Value>,
    pub preprocessing: Vec<Value>,
    #[serde(rename = "nComponents")]
    pub n_components: Vec<i32>,
    pub recipe: Vec<Value>,
    pub candidates: Vec<BTreeMap<String, Value>>,
    pub evaluation: Value,
    pub root_seed: Option<u64>,
}

fn refuse(path: &str, message: impl std::fmt::Display) -> DagMlError {
    DagMlError::GraphValidation(format!("{path}: {message}"))
}

fn keys(object: &Map<String, Value>, allowed: &[&str], path: &str) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) && !matches!(key.as_str(), "_comment" | "comment") {
            return Err(refuse(
                &format!("{path}.{key}"),
                if key.starts_with('_') {
                    "unsupported execution intent"
                } else {
                    "unsupported portable field"
                },
            ));
        }
    }
    Ok(())
}

fn object(value: &Value, path: &str) -> Result<Map<String, Value>> {
    value
        .as_object()
        .cloned()
        .ok_or_else(|| refuse(path, "object required"))
}

fn number(value: &Value, path: &str) -> Result<f64> {
    let n = value
        .as_f64()
        .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()));
    n.filter(|n| n.is_finite())
        .ok_or_else(|| refuse(path, "finite number required"))
}

fn integer(value: &Value, minimum: i32, path: &str) -> Result<i32> {
    let n = number(value, path)?;
    if n.fract() != 0.0 || n < minimum as f64 || n > i32::MAX as f64 {
        return Err(refuse(
            path,
            format!("integer in {minimum}..i32::MAX required"),
        ));
    }
    Ok(n as i32)
}

fn defaults(params: &mut Map<String, Value>, values: Value) {
    for (key, value) in values.as_object().expect("static defaults") {
        params.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

/// Alias interpretation belongs to this owner profile, never to a binding runner.
fn method(class: &str) -> Option<&str> {
    Some(match class {
        "n4m.SNV"
        | "preprocessing.snv"
        | "nirs4all.operators.transforms.SNV"
        | "nirs4all.operators.transforms.StandardNormalVariate"
        | "nirs4all.operators.transforms.scalers.StandardNormalVariate" => {
            "preprocessing.scatter.snv"
        }
        "n4m.SavitzkyGolay"
        | "preprocessing.savgol"
        | "nirs4all.operators.transforms.SavitzkyGolay"
        | "nirs4all.operators.transforms.nirs.SavitzkyGolay" => {
            "preprocessing.derivatives.savitzky_golay"
        }
        "sklearn.cross_decomposition.PLSRegression"
        | "sklearn.cross_decomposition._pls.PLSRegression"
        | "n4m.PLS"
        | "n4m.PLSRegression"
        | "models.pls.pls_fit_simple" => "models.pls.pls_regression",
        "n4m.LSNV" | "preprocessing.local_snv" => "preprocessing.scatter.local_snv",
        "n4m.RNV" | "preprocessing.robust_snv" => "preprocessing.scatter.robust_snv",
        "n4m.AreaNormalization" | "preprocessing.area_normalization" => {
            "preprocessing.scatter.area_normalization"
        }
        "n4m.Detrend" | "preprocessing.detrend" => "preprocessing.baselines.detrend",
        "n4m.MSC" | "preprocessing.msc" => "preprocessing.scatter.msc",
        "n4m.EMSC" | "preprocessing.emsc" => "preprocessing.scatter.emsc",
        "n4m.SPA" => "selection.spa",
        "n4m.Ridge" => "models.regularized.ridge",
        "n4m.RidgePLS" => "models.regularized.ridge_pls",
        "n4m.RobustPLS" => "models.regularized.robust_pls",
        "n4m.CPPLS" => "models.pls.cppls",
        "n4m.SparseSIMPLS" => "models.sparse.sparse_simpls",
        "n4m.ECR" => "models.specialized.ecr",
        "n4m.ContinuumRegression" => "models.regularized.continuum_regression",
        "n4m.MIRPLS" => "models.multiblock.mir_pls",
        "n4m.FusedSparsePLS" => "models.sparse.fused_sparse_pls",
        "n4m.BaggingPLS" => "models.ensembles.bagging_pls",
        "n4m.BoostingPLS" => "models.ensembles.boosting_pls",
        "n4m.RandomSubspacePLS" => "models.ensembles.random_subspace_pls",
        "n4m.NPLS" => "models.specialized.tensor_pls",
        "n4m.MBPLS" => "models.multiblock.mb_pls",
        "n4m.GroupSparsePLS" => "models.sparse.group_sparse_pls",
        _ => class.strip_prefix("n4m:")?,
    })
}

/// Complete no-fit admission, shared by preview, Rust execution, exports and sessions.
pub fn admit_portable_linear(value: &Value) -> Result<PortableLinearAdmission> {
    let root = if value.is_array() {
        json!({"pipeline":value})
    } else {
        value.clone()
    };
    let root = object(&root, "$")?;
    keys(
        &root,
        &["pipeline", "steps", "name", "description", "random_state"],
        "$",
    )?;
    if root.contains_key("pipeline") && root.contains_key("steps") {
        return Err(refuse("$", "ambiguous pipeline and steps"));
    }
    let steps = root
        .get("pipeline")
        .or_else(|| root.get("steps"))
        .and_then(Value::as_array)
        .ok_or_else(|| refuse("$.pipeline", "step array required"))?;
    let mut splitter = None;
    let mut recipe = Vec::new();
    let mut preprocessing = Vec::new();
    let mut candidates = Vec::new();
    let mut components = Vec::new();
    let mut model_seen = false;
    for (index, value) in steps.iter().enumerate() {
        let path = format!("$.pipeline[{index}]");
        if value.is_object()
            && value
                .as_object()
                .unwrap()
                .keys()
                .all(|k| matches!(k.as_str(), "_comment" | "comment"))
        {
            continue;
        }
        if model_seen {
            return Err(refuse(&path, "model must be the final step"));
        }
        let step = object(value, &path)?;
        keys(
            &step,
            &["class", "params", "model", "_range_", "param", "name"],
            &path,
        )?;
        let terminal = step.contains_key("model");
        if terminal && step.contains_key("class") {
            return Err(refuse(&path, "class and model conflict"));
        }
        if terminal && step.contains_key("params") {
            return Err(refuse(
                &path,
                "terminal params must be declared inside model",
            ));
        }
        if !terminal && (step.contains_key("_range_") || step.contains_key("param")) {
            return Err(refuse(
                &path,
                "only terminal component ranges are supported",
            ));
        }
        let spec = if terminal {
            object(&step["model"], &format!("{path}.model"))?
        } else {
            step.clone()
        };
        if terminal {
            keys(&spec, &["class", "params"], &format!("{path}.model"))?;
        }
        let class = spec
            .get("class")
            .and_then(Value::as_str)
            .ok_or_else(|| refuse(&path, "class required"))?;
        let mut params = match spec.get("params") {
            None => Map::new(),
            Some(v) => object(v, &format!("{path}.params"))?,
        };
        params.remove("_comment");
        params.remove("comment");
        if matches!(
            class,
            "n4m.KennardStone"
                | "splitters.kennard_stone"
                | "nirs4all.operators.splitters.KennardStoneSplitter"
                | "nirs4all.operators.splitters.splitters.KennardStoneSplitter"
        ) {
            if terminal || splitter.is_some() {
                return Err(refuse(&path, "one optional splitter required"));
            }
            keys(&params, &["test_size"], &format!("{path}.params"))?;
            let size = number(
                params.get("test_size").unwrap_or(&json!(0.25)),
                &format!("{path}.params.test_size"),
            )?;
            if size <= 0.0 || size >= 1.0 {
                return Err(refuse(&path, "test_size must be between zero and one"));
            }
            splitter = Some(json!({"type":"KennardStone","params":{"test_size":size}}));
            continue;
        }
        let mut id = if class == "n4m.Selector" {
            keys(
                &params,
                &["method", "method_id", "n_components", "method_params"],
                &format!("{path}.params"),
            )?;
            if params.contains_key("method") && params.contains_key("method_id") {
                return Err(refuse(&path, "Selector method and method_id conflict"));
            }
            let method = params
                .remove("method_id")
                .or_else(|| params.remove("method"))
                .and_then(|v| v.as_str().map(str::to_owned))
                .ok_or_else(|| refuse(&path, "Selector method required"))?;
            let count = params
                .remove("n_components")
                .ok_or_else(|| refuse(&path, "Selector n_components required"))?;
            let nested = params.remove("method_params").unwrap_or_else(|| json!({}));
            params = object(&nested, &format!("{path}.params.method_params"))?;
            params.insert("n_components".into(), json!(integer(&count, 1, &path)?));
            if method.starts_with("selection.") {
                method
            } else {
                let name = method
                    .strip_suffix("_select")
                    .ok_or_else(|| refuse(&path, "Selector requires a named native selector"))?;
                format!("selection.{name}")
            }
        } else {
            method(class)
                .ok_or_else(|| refuse(&path, format!("unsupported profile/operator {class}")))?
                .to_string()
        };
        if id == "models.pls.pls_regression"
            && params.get("algo").and_then(Value::as_str) == Some("opls_nipals")
        {
            params.remove("algo");
            id = "models.pls.opls".into();
        }
        if id == "models.pls.pls_regression"
            && params.get("algo").and_then(Value::as_str) == Some("pcr_svd")
        {
            params.remove("algo");
            id = "models.pls.pcr".into();
        }
        match id.as_str() {
            "preprocessing.scatter.snv" => {
                keys(
                    &params,
                    &["with_mean", "with_std", "ddof"],
                    &format!("{path}.params"),
                )?;
                defaults(
                    &mut params,
                    json!({"with_mean":true,"with_std":true,"ddof":0}),
                );
                for key in ["with_mean", "with_std"] {
                    if !params[key].is_boolean() {
                        return Err(refuse(&format!("{path}.params.{key}"), "boolean required"));
                    }
                }
                params["ddof"] =
                    json!(integer(&params["ddof"], 0, &format!("{path}.params.ddof"))?);
            }
            "preprocessing.derivatives.savitzky_golay" => {
                keys(
                    &params,
                    &[
                        "window_length",
                        "window",
                        "polyorder",
                        "deriv",
                        "delta",
                        "mode",
                        "cval",
                    ],
                    &format!("{path}.params"),
                )?;
                if let Some(window) = params.remove("window") {
                    if params.contains_key("window_length") {
                        return Err(refuse(&path, "ambiguous SG window"));
                    }
                    params.insert("window_length".into(), window);
                }
                defaults(
                    &mut params,
                    json!({"window_length":11,"polyorder":3,"deriv":0,"delta":1.0,"mode":"interp","cval":0.0}),
                );
                for (key, min) in [("window_length", 3), ("polyorder", 0), ("deriv", 0)] {
                    params[key] =
                        json!(integer(&params[key], min, &format!("{path}.params.{key}"))?);
                }
                if params["window_length"].as_i64().unwrap() % 2 == 0
                    || params["polyorder"].as_i64() >= params["window_length"].as_i64()
                    || params["deriv"].as_i64() > params["polyorder"].as_i64()
                {
                    return Err(refuse(&path, "invalid SG window/degree/derivative"));
                }
                if number(&params["delta"], &format!("{path}.params.delta"))? != 1.0 {
                    return Err(refuse(
                        &format!("{path}.params.delta"),
                        "only delta=1 is supported",
                    ));
                }
                params["delta"] = json!(1.0);
                params["cval"] = json!(number(&params["cval"], &format!("{path}.params.cval"))?);
                let modes = ["mirror", "constant", "nearest", "wrap", "interp"];
                if let Some(n) = params["mode"].as_u64() {
                    params["mode"] = json!(modes
                        .get(n as usize)
                        .ok_or_else(|| refuse(&path, "invalid SG mode"))?);
                }
                if !params["mode"].as_str().is_some_and(|m| modes.contains(&m)) {
                    return Err(refuse(
                        &format!("{path}.params.mode"),
                        "unsupported SG mode",
                    ));
                }
            }
            "models.pls.pls_regression" => {
                keys(
                    &params,
                    &[
                        "n_components",
                        "solver",
                        "algo",
                        "scale",
                        "center_x",
                        "scale_x",
                        "center_y",
                        "scale_y",
                    ],
                    &format!("{path}.model.params"),
                )?;
                if let Some(scale) = params.remove("scale") {
                    if !scale.is_boolean() {
                        return Err(refuse(
                            &format!("{path}.model.params.scale"),
                            "boolean required",
                        ));
                    }
                    for key in ["scale_x", "scale_y"] {
                        if params.get(key).is_some_and(|v| v != &scale) {
                            return Err(refuse(&path, "conflicting scale and native scale flag"));
                        }
                        params.insert(key.into(), scale.clone());
                    }
                }
                if let Some(algo) = params.remove("algo") {
                    if params.contains_key("solver") {
                        return Err(refuse(&path, "algo and solver conflict"));
                    }
                    let solver = if let Some(value) = algo.as_str() {
                        value
                            .strip_prefix("pls_")
                            .filter(|v| {
                                [
                                    "nipals",
                                    "simpls",
                                    "orthogonal_scores",
                                    "kernel_algorithm",
                                    "wide_kernel",
                                    "svd",
                                    "power",
                                    "randomized_svd",
                                ]
                                .contains(v)
                            })
                            .ok_or_else(|| {
                                refuse(
                                    &format!("{path}.model.params.algo"),
                                    "unsupported PLS algorithm",
                                )
                            })?
                            .to_owned()
                    } else {
                        match algo.as_i64() {
                            Some(0) => "nipals".into(),
                            Some(1) => "simpls".into(),
                            _ => return Err(refuse(&path, "unsupported PLS algorithm")),
                        }
                    };
                    params.insert("solver".into(), json!(solver));
                }
                defaults(
                    &mut params,
                    json!({"n_components":2,"solver":"simpls","center_x":true,"scale_x":true,"center_y":true,"scale_y":true}),
                );
                for key in ["center_x", "scale_x", "center_y", "scale_y"] {
                    if !params[key].is_boolean() {
                        return Err(refuse(
                            &format!("{path}.model.params.{key}"),
                            "boolean required",
                        ));
                    }
                }
                if !params["solver"].as_str().is_some_and(|s| {
                    [
                        "nipals",
                        "simpls",
                        "orthogonal_scores",
                        "kernel_algorithm",
                        "wide_kernel",
                        "svd",
                        "power",
                        "randomized_svd",
                    ]
                    .contains(&s)
                }) {
                    return Err(refuse(
                        &format!("{path}.model.params.solver"),
                        "unsupported PLS solver",
                    ));
                }
            }
            "models.pls.pcr" | "models.pls.opls" => {
                keys(
                    &params,
                    &["n_components", "center_x", "scale_x", "center_y", "scale_y"],
                    &format!("{path}.model.params"),
                )?;
                defaults(
                    &mut params,
                    json!({"n_components":2,"center_x":true,"scale_x":true,"center_y":true,"scale_y":true}),
                );
                for key in ["center_x", "scale_x", "center_y", "scale_y"] {
                    if !params[key].is_boolean() {
                        return Err(refuse(
                            &format!("{path}.model.params.{key}"),
                            "boolean required",
                        ));
                    }
                }
            }
            "preprocessing.scatter.local_snv" => defaults(
                &mut params,
                json!({"window":11,"pad_mode":"reflect","constant_value":0.0}),
            ),
            "preprocessing.scatter.robust_snv" => defaults(
                &mut params,
                json!({"with_center":true,"with_scale":true,"k":1.4826}),
            ),
            "preprocessing.scatter.area_normalization" => {
                defaults(&mut params, json!({"method":"sum"}))
            }
            "preprocessing.baselines.detrend" => defaults(&mut params, json!({"polyorder":1})),
            "preprocessing.scatter.emsc" => defaults(&mut params, json!({"degree":2})),
            "selection.spa" => {
                defaults(&mut params, json!({"n_components":2}));
            }
            "models.regularized.ridge" => defaults(&mut params, json!({"alpha":1.0})),
            _ => {}
        }
        if terminal {
            model_seen = true;
            if id != "models.regularized.ridge" {
                defaults(&mut params, json!({"n_components":2}));
                let base = integer(
                    &params["n_components"],
                    1,
                    &format!("{path}.model.params.n_components"),
                )?;
                params["n_components"] = json!(base);
                components.push(base);
            }
            if let Some(range) = step.get("_range_") {
                if step.get("param").and_then(Value::as_str) != Some("n_components")
                    || id == "models.regularized.ridge"
                {
                    return Err(refuse(&path, "only component ranges supported"));
                }
                let range = range
                    .as_array()
                    .filter(|r| r.len() == 3)
                    .ok_or_else(|| refuse(&path, "range requires [start,stop,step]"))?;
                let start = integer(&range[0], 1, &path)?;
                let stop = integer(&range[1], 1, &path)?;
                let step = integer(&range[2], 1, &path)?;
                if start > stop {
                    return Err(refuse(&path, "range start exceeds stop"));
                }
                let count = (i64::from(stop) - i64::from(start)) / i64::from(step) + 1;
                if count > 10000 {
                    return Err(refuse(&path, "component range exceeds 10000 candidates"));
                }
                components = (0..count)
                    .map(|i| (i64::from(start) + i * i64::from(step)) as i32)
                    .collect();
            } else if step.contains_key("param") {
                return Err(refuse(&path, "param requires _range_"));
            }
            if components.is_empty() {
                candidates.push(params.clone().into_iter().collect());
            } else {
                for n in &components {
                    let mut p = params.clone();
                    p.insert("n_components".into(), json!(n));
                    candidates.push(p.into_iter().collect());
                }
            }
        } else {
            let kind = match id.as_str() {
                "preprocessing.scatter.snv" => "StandardNormalVariate",
                "preprocessing.derivatives.savitzky_golay" => "SavitzkyGolay",
                other => other,
            };
            preprocessing.push(json!({"type":kind,"params":params}));
        }
        recipe.push(json!({"methodId":id,"params":params}));
    }
    if !model_seen {
        return Err(refuse("$.pipeline", "one terminal regressor required"));
    }
    let root_seed = root
        .get("random_state")
        .map(|v| {
            v.as_u64()
                .ok_or_else(|| refuse("$.random_state", "nonnegative integer required"))
        })
        .transpose()?;
    let scope = if splitter.is_some() {
        "selection_validation"
    } else {
        "training"
    };
    Ok(PortableLinearAdmission {
        schema_version: 1,
        profile: PORTABLE_LINEAR_PROFILE.into(),
        owner_profile: "pipeline_steps.v1".into(),
        mode: if splitter.is_some() {
            "single_holdout"
        } else {
            "training"
        }
        .into(),
        name: root
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("pipeline")
            .into(),
        splitter,
        preprocessing,
        n_components: components,
        recipe,
        candidates,
        evaluation: json!({"scope":scope,"independent_test":false}),
        root_seed,
    })
}

/// Compile into the existing DSL bank shape, with IO facts and optional Methods KS FoldSet.
pub fn compile_portable_linear(
    admission: &PortableLinearAdmission,
    binding: DataBinding,
    fold_set: Option<crate::FoldSet>,
    registry: &ControllerRegistry,
) -> Result<CompiledPipelineDsl> {
    let target = binding.node_id.clone();
    let dimensions = json!([
        {"name":"chain","choices":[{"label":"chain:0","value":0,"param_overrides":[{"node_id":target,"params":{"pipeline_steps_v1":admission.recipe}}]}]},
        {"name":"model_parameters","choices":admission.candidates.iter().enumerate().map(|(i,p)|json!({"label":format!("candidate:{i}"),"value":i,"param_overrides":[{"node_id":target,"params":p}]})).collect::<Vec<_>>()}]);
    let policy = json!({"split_unit":"sample","forbid_origin_cross_fold":true,"allow_observation_split_with_shared_target":false,"require_group_ids":false,"unsafe_flags":[]});
    let spec: PipelineDslSpec = serde_json::from_value(
        json!({"id":"portable:linear","campaign_id":"portable:campaign","root_seed":admission.root_seed,
        "input":{"name":"x","representation":"tabular_numeric"},"generation_strategy":"cartesian","max_variants":admission.candidates.len(),"generation_dimensions":dimensions,
        "leakage_policy":policy,"data_bindings":[binding],
        "split_invocation":fold_set.map(|fold|json!({"id":"portable:ks","controller_id":null,"params":{},"fold_set":fold,"leakage_policy":policy})),
        "steps":[{"kind":"model","id":target,"operator":{"type":"N4mRolePipeline"},"params":{"pipeline_steps_v1":admission.recipe}}]}),
    )?;
    compile_pipeline_dsl_with_generation_and_controller_registry(&spec, registry)
}

#[cfg(test)]
mod portable_tests {
    use super::*;
    #[test]
    fn flags_survive_ranges_and_four_b02_plans_remain_distinct() {
        let mut plans = Vec::new();
        for std in [false, true] {
            for scale in [false, true] {
                let value = json!({"pipeline":[{"class":"n4m.SNV","params":{"with_std":std}},{"model":{"class":"n4m.PLS","params":{"n_components":5,"scale":scale}},"_range_":[1,3,1],"param":"n_components"}]});
                let p = admit_portable_linear(&value).unwrap();
                assert_eq!(p.n_components, vec![1, 2, 3]);
                assert_eq!(p.recipe[0]["params"]["with_std"], json!(std));
                assert_eq!(p.candidates[0]["scale_x"], json!(scale));
                assert!(!p.candidates[0].contains_key("scale"));
                plans.push(serde_json::to_string(&p).unwrap());
            }
        }
        assert_eq!(plans.into_iter().collect::<BTreeSet<_>>().len(), 4);
    }
    #[test]
    fn unsupported_intent_and_parameters_are_refused_without_operators() {
        for v in [
            json!({"cv":3,"pipeline":[{"model":{"class":"n4m.PLS"}}]}),
            json!({"pipeline":[{"class":"n4m.SNV","params":{"unknown":true}},{"model":{"class":"n4m.PLS"}}]}),
            json!({"pipeline":[{"class":"n4m.KennardStone","params":{"n_train":5}},{"model":{"class":"n4m.PLS"}}]}),
            json!({"pipeline":[{"model":{"class":"n4m.PLS"},"params":{"scale":false}}]}),
        ] {
            assert!(admit_portable_linear(&v).is_err());
        }
    }
    #[test]
    fn opls_maps_only_to_the_exact_methods_role_and_five_parameters() {
        let mut definition = json!({"pipeline":[{"model":{"class":"n4m.PLS","params":{"algo":"opls_nipals","n_components":3,"scale_x":false,"scale_y":false}}}]});
        let admitted = admit_portable_linear(&definition).unwrap();
        assert_eq!(admitted.recipe[0]["methodId"], "models.pls.opls");
        assert_eq!(admitted.candidates[0].len(), 5);
        assert_eq!(admitted.candidates[0]["n_components"], 3);
        assert_eq!(admitted.candidates[0]["scale_x"], false);
        assert_eq!(admitted.candidates[0]["center_y"], true);
        assert!(!admitted.candidates[0].contains_key("solver"));
        definition["pipeline"][0]["model"]["params"]["solver"] = json!("nipals");
        assert!(admit_portable_linear(&definition).is_err());
    }
}
