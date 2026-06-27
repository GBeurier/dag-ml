use super::*;
use crate::controller::{
    ArtifactPolicy, ControllerCapability, ControllerFitScope, ControllerManifest, OperatorSelector,
    RngPolicy,
};
use crate::phase::Phase;

fn registry_manifest(id: &str, kind: NodeKind, aliases: &[&str]) -> ControllerManifest {
    ControllerManifest {
        controller_id: crate::ids::ControllerId::new(id).unwrap(),
        controller_version: "0.1.0".to_string(),
        operator_kind: kind,
        priority: 0,
        supported_phases: BTreeSet::from([Phase::FitCv]),
        input_ports: Vec::new(),
        output_ports: Vec::new(),
        data_requirements: None,
        capabilities: BTreeSet::from([ControllerCapability::Deterministic]),
        operator_selectors: vec![OperatorSelector {
            aliases: aliases.iter().map(|alias| (*alias).to_string()).collect(),
            ..OperatorSelector::default()
        }],
        fit_scope: ControllerFitScope::FoldTrain,
        rng_policy: RngPolicy::UsesCoreSeed,
        artifact_policy: ArtifactPolicy::Serializable,
    }
}

#[test]
fn compiles_linear_pipeline_dsl_to_valid_graph() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-linear-smoke",
  "steps": [
    {
      "kind": "transform",
      "id": "transform:snv",
      "operator": {"type": "StandardNormalVariate"},
      "seed_label": "snv"
    },
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "RandomForestRegressor"},
      "params": {"n_estimators": 100},
      "seed_label": "base"
    }
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();

    assert_eq!(graph.id, "dsl-linear-smoke");
    assert_eq!(graph.nodes.len(), 2);
    assert_eq!(graph.edges.len(), 1);
    assert_eq!(graph.nodes[0].kind, NodeKind::Transform);
    assert_eq!(graph.nodes[1].kind, NodeKind::Model);
    assert_eq!(graph.edges[0].source.node_id.as_str(), "transform:snv");
    assert_eq!(graph.edges[0].target.node_id.as_str(), "model:base");
    assert_eq!(graph.edges[0].contract.kind, PortKind::Data);
    graph.validate().unwrap();
}

#[test]
fn compiles_pipeline_dsl_unit_contracts_to_graph_interface() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-unit-contract-smoke",
  "input": {
    "name": "spectra",
    "representation": "tabular",
    "unit_level": "observation",
    "alignment_key": "sample_id",
    "target_level": "physical_sample"
  },
  "output": {
    "name": "prediction",
    "representation": "regression",
    "unit_level": "physical_sample",
    "alignment_key": "sample_id",
    "target_level": "physical_sample"
  },
  "steps": [
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "RandomForestRegressor"}
    }
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();

    assert_eq!(
        graph.interface.inputs[0].unit_level,
        Some(EntityUnitLevel::Observation)
    );
    assert_eq!(
        graph.interface.inputs[0].alignment_key.as_deref(),
        Some("sample_id")
    );
    assert_eq!(
        graph.interface.outputs[0].unit_level,
        Some(EntityUnitLevel::PhysicalSample)
    );
    assert_eq!(
        graph.interface.outputs[0].representation.as_deref(),
        Some("regression")
    );
}

#[test]
fn compiles_branch_merge_predictions_plus_original_dsl() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-branch-merge-smoke",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "b0",
          "steps": [
            {
              "kind": "model",
              "id": "branch:b0.model:ridge",
              "operator": {"type": "Ridge"},
              "params": {"alpha": 0.3},
              "seed_label": "branch:b0"
            }
          ]
        },
        {
          "id": "b1",
          "steps": [
            {
              "kind": "augmentation",
              "id": "branch:b1.augment:noise",
              "operator": {"type": "GaussianNoise"},
              "params": {"scope": "train_only"},
              "seed_label": "branch:b1.augment",
              "shape": {
                "fit_rows": "fold_train",
                "predict_rows": "fold_validation",
                "augmentation_policy": {
                  "sample_scope": "train_only",
                  "feature_scope": "none",
                  "require_origin_id": true,
                  "inherit_group": true,
                  "inherit_target": true
                }
              }
            },
            {
              "kind": "model",
              "id": "branch:b1.model:rf",
              "operator": {"type": "RandomForestRegressor"},
              "params": {"n_estimators": 64},
              "seed_label": "branch:b1"
            }
          ]
        }
      ]
    },
    {
      "kind": "merge_model",
      "id": "merge:stack.pred_plus_original.meta:ridge",
      "operator": {"type": "RidgeMetaStacker"},
      "params": {"alpha": 0.2},
      "seed_label": "merge:stack"
    }
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();

    assert_eq!(graph.nodes.len(), 4);
    assert_eq!(graph.edges.len(), 3);
    let merge = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "merge:stack.pred_plus_original.meta:ridge")
        .unwrap();
    assert_eq!(merge.ports.inputs.len(), 3);
    assert_eq!(merge.ports.inputs[0].name, "b0_oof");
    assert_eq!(merge.ports.inputs[1].name, "b1_oof");
    assert_eq!(merge.ports.inputs[2].name, "x_original");
    let prediction_edges = graph
        .edges
        .iter()
        .filter(|edge| edge.contract.kind == PortKind::Prediction)
        .collect::<Vec<_>>();
    assert_eq!(prediction_edges.len(), 2);
    assert!(prediction_edges
        .iter()
        .all(|edge| edge.contract.requires_oof));
    assert!(prediction_edges
        .iter()
        .all(|edge| edge.contract.requires_fold_alignment));
    assert!(graph.edges.iter().any(|edge| edge.source.node_id.as_str()
        == "branch:b1.augment:noise"
        && edge.target.node_id.as_str() == "branch:b1.model:rf"));
    graph.validate().unwrap();
}

#[test]
fn compiles_separation_branch_view_plans() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-separation-branch-views",
  "steps": [
    {
      "kind": "branch",
      "mode": "by_metadata",
      "selector": {"metadata_key": "site"},
      "branches": [
        {
          "id": "site_a",
          "selector": "A",
          "steps": [
            {"kind": "model", "id": "model:site.a", "operator": {"type": "PLSRegression"}}
          ]
        },
        {
          "id": "site_b",
          "selector": {"value": "B"},
          "steps": [
            {"kind": "model", "id": "model:site.b", "operator": {"type": "Ridge"}}
          ]
        }
      ]
    },
    {
      "kind": "merge_model",
      "id": "model:site.meta",
      "operator": {"type": "Ridge"},
      "include_original_data": false
    }
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

    assert_eq!(compiled.branch_view_plans.len(), 2);
    assert_eq!(
        compiled.campaign_template.branch_view_plans,
        compiled.branch_view_plans
    );
    assert_eq!(
        compiled.branch_view_plans[0].mode,
        BranchViewMode::ByMetadata
    );
    assert_eq!(compiled.branch_view_plans[0].selector.metadata["site"], "A");
    assert_eq!(compiled.branch_view_plans[1].selector.metadata["site"], "B");
    let site_model = compiled
        .graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:site.a")
        .unwrap();
    assert_eq!(
        site_model.metadata["dsl_branch_view_plan"]["selector"]["metadata"]["site"],
        "A"
    );
}

#[test]
fn refuses_separation_branch_without_selector() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-bad-separation-branch",
  "steps": [
    {
      "kind": "branch",
      "mode": "by_source",
      "branches": [
        {
          "id": "nir",
          "steps": [
            {"kind": "model", "id": "model:nir", "operator": {"type": "Ridge"}}
          ]
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl_with_generation(&spec)
        .unwrap_err()
        .to_string();

    assert!(error.contains("by_source branch `nir` requires a selector"));
}

#[test]
fn compiles_branch_feature_merge_into_downstream_model() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-branch-feature-merge",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "snv",
          "steps": [
            {
              "kind": "transform",
              "id": "branch:snv.transform",
              "operator": {"type": "SNV"}
            }
          ]
        },
        {
          "id": "msc",
          "steps": [
            {
              "kind": "transform",
              "id": "branch:msc.transform",
              "operator": {"type": "MSC"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:features",
      "merge_mode": "features",
      "output_as": "features",
      "include_original_data": false
    },
    {
      "kind": "model",
      "id": "model:pls",
      "operator": {"type": "PLSRegression"}
    }
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    let merge = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "merge:features")
        .unwrap();
    assert_eq!(merge.kind, NodeKind::FeatureJoin);
    assert_eq!(merge.ports.inputs.len(), 2);
    assert!(merge.ports.inputs.iter().any(|port| port.name == "snv_x"));
    assert!(merge.ports.inputs.iter().any(|port| port.name == "msc_x"));
    assert!(graph.edges.iter().any(|edge| {
        edge.source.node_id.as_str() == "branch:snv.transform"
            && edge.target.node_id.as_str() == "merge:features"
            && edge.target.port_name == "snv_x"
            && edge.contract.kind == PortKind::Data
    }));
    assert!(graph.edges.iter().any(|edge| {
        edge.source.node_id.as_str() == "merge:features"
            && edge.target.node_id.as_str() == "model:pls"
            && edge.contract.kind == PortKind::Data
    }));
    assert!(!graph
        .edges
        .iter()
        .any(|edge| edge.contract.kind == PortKind::Prediction));
}

#[test]
fn compiles_nirs4all_style_multi_model_branch_and_separate_merge() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-nirs4all-branch-parity",
  "steps": [
    {
      "kind": "branch",
      "mode": "duplication",
      "selector": {"scope": "all_samples"},
      "branches": [
        {
          "id": "pls_path",
          "steps": [
            {
              "kind": "model",
              "id": "branch:pls.model:pls5",
              "operator": {"class": "sklearn.cross_decomposition.PLSRegression"},
              "params": {"n_components": 5}
            },
            {
              "kind": "model",
              "id": "branch:pls.model:pls10",
              "operator": {"class": "sklearn.cross_decomposition.PLSRegression"},
              "params": {"n_components": 10}
            }
          ]
        },
        {
          "id": "rf_path",
          "selector": {"source": "nir"},
          "steps": [
            {
              "kind": "transform",
              "id": "branch:rf.transform:snv",
              "operator": {"class": "nirs4all.operators.transforms.StandardNormalVariate"}
            },
            {
              "kind": "model",
              "id": "branch:rf.model:rf",
              "operator": {"class": "sklearn.ensemble.RandomForestRegressor"},
              "params": {"n_estimators": 64}
            },
            {
              "kind": "model",
              "id": "branch:rf.model:gbr",
              "operator": {"class": "sklearn.ensemble.GradientBoostingRegressor"},
              "params": {"n_estimators": 32}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:stack.predictions_plus_original",
      "merge_mode": "predictions_plus_original",
      "output_as": "features",
      "include_original_data": true,
      "selectors": [
        {"branch": "pls_path", "select": "best", "metric": "rmse"},
        {"branch": "rf_path", "select": {"top_k": 2}, "metric": "r2"}
      ],
      "metadata": {"on_missing": "warn"}
    },
    {
      "kind": "model",
      "id": "model:meta.ridge",
      "operator": {"class": "sklearn.linear_model.Ridge"},
      "variants": [
        {"label": "low", "params": {"alpha": 0.1}},
        {"label": "mid", "params": {"alpha": 0.5}}
      ]
    },
    {
      "kind": "model",
      "id": "model:meta.rf",
      "operator": {"class": "sklearn.ensemble.RandomForestRegressor"},
      "params": {"n_estimators": 30}
    }
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
    let graph = compiled.graph;
    let merge = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "merge:stack.predictions_plus_original")
        .unwrap();

    assert_eq!(merge.kind, NodeKind::MixedJoin);
    assert_eq!(merge.ports.inputs.len(), 5);
    assert_eq!(merge.ports.outputs[0].kind, PortKind::Data);
    assert_eq!(merge.metadata["merge_mode"], "predictions_plus_original");
    assert_eq!(merge.metadata["selectors"][0]["branch"], "pls_path");
    let rf_model = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "branch:rf.model:rf")
        .unwrap();
    assert_eq!(rf_model.metadata["dsl_branch"], "rf_path");
    assert_eq!(rf_model.metadata["dsl_branch_mode"], "duplication");
    assert_eq!(
        rf_model.metadata["dsl_branch_step_selector"]["scope"],
        "all_samples"
    );
    assert_eq!(rf_model.metadata["dsl_branch_selector"]["source"], "nir");
    assert_eq!(
        graph
            .edges
            .iter()
            .filter(|edge| edge.target.node_id == merge.id
                && edge.contract.kind == PortKind::Prediction
                && edge.contract.requires_oof)
            .count(),
        4
    );
    assert!(graph
        .edges
        .iter()
        .any(|edge| edge.source.node_id == merge.id
            && edge.target.node_id.as_str() == "model:meta.ridge"
            && edge.contract.kind == PortKind::Data));
    assert!(graph
        .edges
        .iter()
        .any(|edge| edge.source.node_id == merge.id
            && edge.target.node_id.as_str() == "model:meta.rf"
            && edge.contract.kind == PortKind::Data));
    assert_eq!(compiled.generation.dimensions.len(), 1);
    assert_eq!(
        compiled.generation.dimensions[0].name,
        "model:meta.ridge.params"
    );
    graph.validate().unwrap();
}

#[test]
fn merge_selectors_reject_unknown_branch_and_missing_metric() {
    let unknown_branch: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-bad-merge-selector-branch",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "known",
          "steps": [
            {
              "kind": "model",
              "id": "branch:known.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:bad.selector",
      "selectors": [
        {"branch": "missing", "select": "all"}
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let error = compile_pipeline_dsl_with_generation(&unknown_branch).unwrap_err();
    assert!(format!("{error}").contains("does not match any pending prediction input"));

    let missing_metric: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-bad-merge-selector-metric",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "known",
          "steps": [
            {
              "kind": "model",
              "id": "branch:known.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:bad.metric",
      "selectors": [
        {"branch": "known", "select": "best"}
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let error = compile_pipeline_dsl_with_generation(&missing_metric).unwrap_err();
    assert!(format!("{error}").contains("requires a non-empty metric"));
}

#[test]
fn merge_selectors_reject_top_k_above_scope() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-bad-merge-selector-top-k",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "known",
          "steps": [
            {
              "kind": "model",
              "id": "branch:known.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:bad.topk",
      "selectors": [
        {"branch": "known", "select": {"top_k": 2}, "metric": "rmse"}
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
    assert!(format!("{error}").contains("top_k=2 exceeds 1 matched prediction inputs"));
}

#[test]
fn compiles_nirs4all_shape_changing_and_tuning_surface() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-nirs4all-shape-parity",
  "steps": [
    {
      "kind": "y_transform",
      "id": "target:scale",
      "operator": {"class": "sklearn.preprocessing.StandardScaler"}
    },
    {
      "kind": "tag",
      "id": "tag:y_outliers",
      "operator": {"class": "nirs4all.filters.YOutlierFilter"},
      "params": {"method": "iqr"}
    },
    {
      "kind": "exclude",
      "id": "exclude:train_outliers",
      "operator": {"class": "nirs4all.filters.YOutlierFilter"},
      "params": {"mode": "any"}
    },
    {
      "kind": "sample_augmentation",
      "id": "augment:sample.noise",
      "operator": {"class": "nirs4all.operators.transforms.GaussianAdditiveNoise"},
      "params": {"count": 3, "selection": "random"},
      "shape": {
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "augmentation_policy": {
          "sample_scope": "train_only",
          "feature_scope": "none",
          "require_origin_id": true,
          "inherit_group": true,
          "inherit_target": true
        }
      }
    },
    {
      "kind": "feature_augmentation",
      "id": "augment:feature.views",
      "operator": {"class": "nirs4all.operators.transforms.FeatureAugmentation"},
      "params": {"action": "extend"},
      "shape": {
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "feature_namespace": "augmented_views",
        "augmentation_policy": {
          "sample_scope": "none",
          "feature_scope": "train_only",
          "require_origin_id": false
        }
      }
    },
    {
      "kind": "concat_transform",
      "id": "join:concat.multi_view",
      "branches": [
        {
          "id": "pca",
          "steps": [
            {
              "id": "concat:pca",
              "operator": {"class": "sklearn.decomposition.PCA"},
              "params": {"n_components": 20}
            }
          ]
        },
        {
          "id": "derivative_pca",
          "steps": [
            {
              "id": "concat:derivative",
              "operator": {"class": "nirs4all.operators.transforms.FirstDerivative"}
            },
            {
              "id": "concat:derivative.pca",
              "operator": {"class": "sklearn.decomposition.PCA"},
              "params": {"n_components": 10}
            }
          ]
        }
      ],
      "shape": {
        "fit_rows": "fold_train",
        "feature_namespace": "concat.multi_view",
        "selection_policy": {
          "scope": "unsupervised"
        }
      }
    },
    {
      "kind": "model",
      "id": "model:tuned",
      "operator": {"class": "sklearn.ensemble.RandomForestRegressor"},
      "finetune_params": {
        "n_trials": 10,
        "approach": "single",
        "eval_mode": "mean",
        "sampler": "random",
        "metric": "rmse",
        "model_params": {
          "max_depth": [3, 5, 7]
        }
      },
      "train_params": {
        "sample_weight": "balanced"
      }
    }
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
    let graph = compiled.graph;
    let kinds = graph
        .nodes
        .iter()
        .map(|node| node.kind.clone())
        .collect::<Vec<_>>();
    assert!(kinds.contains(&NodeKind::YTransform));
    assert!(kinds.contains(&NodeKind::Tag));
    assert!(kinds.contains(&NodeKind::Exclude));
    assert!(kinds.contains(&NodeKind::Augmentation));
    assert!(kinds.contains(&NodeKind::FeatureJoin));
    assert_eq!(compiled.shape_plans.len(), 3);

    let sample_aug = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "augment:sample.noise")
        .unwrap();
    assert_eq!(sample_aug.metadata["dsl_augmentation_kind"], "sample");
    let feature_aug = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "augment:feature.views")
        .unwrap();
    assert_eq!(feature_aug.metadata["dsl_augmentation_kind"], "feature");
    let model = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:tuned")
        .unwrap();
    assert_eq!(model.metadata["dsl_tuning"]["n_trials"], 10);
    assert_eq!(
        model.metadata["dsl_train_params"]["sample_weight"],
        "balanced"
    );
    graph.validate().unwrap();
}

#[test]
fn extracts_node_param_variants_into_generation_spec() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-generation-smoke",
  "max_variants": 4,
  "steps": [
    {
      "kind": "transform",
      "id": "transform:preprocess",
      "operator": {"type": "Preprocess"},
      "variants": [
        {
          "label": "snv",
          "params": {"method": "snv"}
        },
        {
          "label": "msc",
          "params": {"method": "msc"}
        }
      ]
    },
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"},
      "variants": [
        {
          "label": "low",
          "params": {"alpha": 0.1}
        },
        {
          "label": "high",
          "params": {"alpha": 1.0}
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

    assert_eq!(compiled.generation.strategy, GenerationStrategy::Cartesian);
    assert_eq!(compiled.generation.max_variants, Some(4));
    assert_eq!(compiled.generation.dimensions.len(), 2);
    assert_eq!(
        compiled.generation.dimensions[0].name,
        "transform:preprocess.params"
    );
    assert_eq!(compiled.generation.dimensions[0].choices[0].label, "snv");
    assert_eq!(
        compiled.generation.dimensions[0].choices[0].param_overrides[0].node_id,
        NodeId::new("transform:preprocess").unwrap()
    );
    assert_eq!(
        compiled.generation.dimensions[1].choices[1].param_overrides[0].params["alpha"],
        1.0
    );
    assert!(compiled.generation_fingerprint.is_some());
    assert_eq!(
        compiled.graph.search_space_fingerprint,
        compiled.generation_fingerprint
    );
    compiled.graph.validate().unwrap();
}

#[test]
fn expands_compact_param_generators_into_generation_dimensions() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-compact-generation",
  "steps": [
    {
      "kind": "model",
      "id": "model:tuned",
      "operator": {"type": "TunedModel"},
      "generators": [
        {
          "kind": "or",
          "name": "model_family",
          "param": "family",
          "values": [
            {"label": "ridge", "value": "ridge"},
            {"label": "rf", "value": "random_forest"}
          ]
        },
        {
          "kind": "range",
          "param": "alpha",
          "start": 0.1,
          "stop": 0.9,
          "step": 0.4
        },
        {
          "kind": "log_range",
          "param": "lambda",
          "start": 0.01,
          "stop": 1.0,
          "count": 3
        },
        {
          "kind": "grid",
          "name": "tree_grid",
          "params": {
            "max_depth": [3, 5],
            "n_estimators": [50, 100]
          },
          "count": 3
        },
        {
          "kind": "pick",
          "param": "views",
          "values": ["snv", "msc", "derivative"],
          "sizes": [1, 2],
          "count": 4
        },
        {
          "kind": "arrange",
          "param": "chain",
          "values": ["snv", "pca", "pls"],
          "sizes": [2],
          "count": 3
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

    assert_eq!(compiled.generation.strategy, GenerationStrategy::Cartesian);
    assert_eq!(compiled.generation.dimensions.len(), 6);
    assert_eq!(compiled.generation.dimensions[0].name, "model_family");
    assert_eq!(compiled.generation.dimensions[0].choices.len(), 2);
    assert_eq!(
        compiled.generation.dimensions[1].name,
        "model:tuned.alpha.range"
    );
    assert_eq!(compiled.generation.dimensions[1].choices.len(), 3);
    assert_eq!(
        compiled.generation.dimensions[1].choices[1].param_overrides[0].params["alpha"],
        0.5
    );
    assert_eq!(
        compiled.generation.dimensions[2].name,
        "model:tuned.lambda.log_range"
    );
    assert_eq!(compiled.generation.dimensions[2].choices.len(), 3);
    assert_eq!(compiled.generation.dimensions[3].name, "tree_grid");
    assert_eq!(compiled.generation.dimensions[3].choices.len(), 3);
    assert_eq!(
        compiled.generation.dimensions[3].choices[2].param_overrides[0].params["n_estimators"],
        50
    );
    assert_eq!(
        compiled.generation.dimensions[4].choices[3].param_overrides[0].params["views"],
        serde_json::json!(["snv", "msc"])
    );
    assert_eq!(
        compiled.generation.dimensions[5].choices[2].param_overrides[0].params["chain"],
        serde_json::json!(["pca", "snv"])
    );
    assert!(compiled.generation_fingerprint.is_some());
}

fn log_range_repro_spec() -> PipelineDslSpec {
    serde_json::from_str(
            r#"{
  "id": "dsl-log-range-fingerprint",
  "steps": [
    {
      "kind": "model",
      "id": "model:tuned",
      "operator": {"type": "TunedModel"},
      "generators": [
        {"kind": "log_range", "param": "lambda", "start": 0.001, "stop": 1.0, "count": 4, "base": 10.0}
      ]
    }
  ]
}"#,
        )
        .unwrap()
}

/// Regression for the native `log_range` generator: the generated value text
/// must be a JSON round-trip fixpoint so the graph `search_space_fingerprint`
/// and the campaign generation spec stay in agreement through the
/// compile -> serialize -> plan boundary, and the four base-10 points expand
/// to 0.001 / 0.01 / 0.1 / 1.0.
#[test]
fn log_range_generator_compiles_and_plans_through_json_roundtrip() {
    let compiled = compile_pipeline_dsl_with_generation(&log_range_repro_spec()).unwrap();

    // The expanded points are the four base-10 log-range positions
    // (0.001 / 0.01 / 0.1 / 1.0, up to floating-point interpolation error).
    let dimension = &compiled.generation.dimensions[0];
    assert_eq!(dimension.name, "model:tuned.lambda.log_range");
    let values = dimension
        .choices
        .iter()
        .map(|choice| choice.value["lambda"].as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(values.len(), 4);
    for (got, want) in values.iter().zip([0.001, 0.01, 0.1, 1.0]) {
        assert!(
            (got - want).abs() <= want * 1e-12,
            "log_range point {got} not close to {want}"
        );
    }

    // In-memory the two fingerprints already agree.
    assert_eq!(
        compiled.graph.search_space_fingerprint,
        compiled.generation_fingerprint
    );

    // JSON round-trip the artifact exactly like the CLI / Python (C-ABI)
    // boundary does, then recompute the campaign-side fingerprint: it must
    // still equal the graph-side fingerprint stored at compile time.
    let graph: GraphSpec =
        serde_json::from_str(&serde_json::to_string(&compiled.graph).unwrap()).unwrap();
    let campaign: crate::plan::CampaignSpec =
        serde_json::from_str(&serde_json::to_string(&compiled.campaign_template).unwrap()).unwrap();
    let expected = graph.search_space_fingerprint.clone().unwrap();
    let actual = generation_spec_fingerprint(&campaign.generation).unwrap();
    assert_eq!(
        expected, actual,
        "log_range search_space_fingerprint must survive the JSON round-trip"
    );

    // And it actually plans (this is the call that previously raised
    // `search_space_fingerprint does not match campaign generation spec`).
    let mut registry = ControllerRegistry::new();
    registry
        .register(registry_manifest(
            "controller:model",
            NodeKind::Model,
            &["TunedModel"],
        ))
        .unwrap();
    let plan =
        crate::plan::build_execution_plan("plan:log-range", graph, campaign, &registry).unwrap();
    assert_eq!(plan.variants.len(), 4);
}

/// The compiled log_range plan must be byte-identical for the same spec
/// (deterministic generation + fingerprints), while `range` and `grid`
/// generators are unaffected by the canonicalization.
#[test]
fn log_range_generation_is_deterministic_and_range_grid_unaffected() {
    let a = compile_pipeline_dsl_with_generation(&log_range_repro_spec()).unwrap();
    let b = compile_pipeline_dsl_with_generation(&log_range_repro_spec()).unwrap();
    assert_eq!(
        serde_json::to_string(&a).unwrap(),
        serde_json::to_string(&b).unwrap(),
    );
    assert_eq!(a.generation_fingerprint, b.generation_fingerprint);

    // range + grid still compile to matching graph/campaign fingerprints
    // (canonicalization is a no-op for their round-trip-stable values).
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-range-grid-fingerprint",
  "steps": [
    {
      "kind": "model",
      "id": "model:tuned",
      "operator": {"type": "TunedModel"},
      "generators": [
        {"kind": "range", "param": "alpha", "start": 0.1, "stop": 0.9, "step": 0.4},
        {"kind": "grid", "name": "tree_grid", "params": {"max_depth": [3, 5]}}
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
    let graph: GraphSpec =
        serde_json::from_str(&serde_json::to_string(&compiled.graph).unwrap()).unwrap();
    let campaign: crate::plan::CampaignSpec =
        serde_json::from_str(&serde_json::to_string(&compiled.campaign_template).unwrap()).unwrap();
    assert_eq!(
        graph.search_space_fingerprint.unwrap(),
        generation_spec_fingerprint(&campaign.generation).unwrap()
    );
}

#[test]
fn compact_param_generators_reject_invalid_counts() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-bad-compact-generation",
  "steps": [
    {
      "kind": "model",
      "id": "model:bad",
      "operator": {"type": "Ridge"},
      "generators": [
        {
          "kind": "or",
          "param": "alpha",
          "values": [0.1, 1.0],
          "count": 0
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
    assert!(format!("{error}").contains("count=0"));
}

#[test]
fn compiles_coordinated_generation_dimensions() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-coordinated-generation",
  "max_variants": 2,
  "generation_dimensions": [
    {
      "name": "stack_profile",
      "choices": [
        {
          "label": "linear_stack",
          "param_overrides": [
            {"node_id": "branch:b0.model:ridge", "params": {"alpha": 0.1}},
            {"node_id": "branch:b1.model:rf", "params": {"max_depth": 4}},
            {"node_id": "merge:stack.pred_plus_original.meta:ridge", "params": {"alpha": 0.05}}
          ]
        },
        {
          "label": "robust_stack",
          "param_overrides": [
            {"node_id": "branch:b0.model:ridge", "params": {"alpha": 1.0}},
            {"node_id": "branch:b1.model:rf", "params": {"max_depth": 8}},
            {"node_id": "merge:stack.pred_plus_original.meta:ridge", "params": {"alpha": 0.5}}
          ]
        }
      ]
    }
  ],
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "b0",
          "steps": [
            {
              "kind": "model",
              "id": "branch:b0.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        },
        {
          "id": "b1",
          "steps": [
            {
              "kind": "model",
              "id": "branch:b1.model:rf",
              "operator": {"type": "RandomForestRegressor"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge_model",
      "id": "merge:stack.pred_plus_original.meta:ridge",
      "operator": {"type": "RidgeMetaStacker"}
    }
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

    assert_eq!(compiled.generation.strategy, GenerationStrategy::Cartesian);
    assert_eq!(compiled.generation.max_variants, Some(2));
    assert_eq!(compiled.generation.dimensions.len(), 1);
    assert_eq!(compiled.generation.dimensions[0].name, "stack_profile");
    assert_eq!(
        compiled.generation.dimensions[0].choices[0]
            .param_overrides
            .len(),
        3
    );
    assert_eq!(
        compiled.generation.dimensions[0].choices[1].param_overrides[2].node_id,
        NodeId::new("merge:stack.pred_plus_original.meta:ridge").unwrap()
    );
    assert_eq!(
        compiled.generation.dimensions[0].choices[1].value
            ["merge:stack.pred_plus_original.meta:ridge"]["alpha"],
        0.5
    );
    assert_eq!(
        compiled.graph.search_space_fingerprint,
        compiled.generation_fingerprint
    );
    compiled.graph.validate().unwrap();
}

#[test]
fn refuses_coordinated_generation_for_unknown_node() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-bad-generation-target",
  "generation_dimensions": [
    {
      "name": "bad_target",
      "choices": [
        {
          "label": "bad",
          "param_overrides": [
            {"node_id": "model:missing", "params": {"alpha": 0.1}}
          ]
        }
      ]
    }
  ],
  "steps": [
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"}
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
    assert!(format!("{error}").contains("references unknown node `model:missing`"));
}

#[test]
fn artifact_contains_campaign_template_without_split_graph_nodes() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-campaign-template",
  "campaign_id": "campaign:dsl.template",
  "root_seed": 123,
  "leakage_policy": {
    "split_unit": "group",
    "require_group_ids": true
  },
  "split_invocation": {
    "id": "split:group-kfold",
    "leakage_policy": {
      "split_unit": "group",
      "require_group_ids": true
    },
    "params": {
      "n_splits": 3
    }
  },
  "generation_dimensions": [
    {
      "name": "model_family",
      "choices": [
        {
          "label": "ridge_low",
          "param_overrides": [
            {"node_id": "model:base", "params": {"alpha": 0.1}}
          ]
        },
        {
          "label": "ridge_high",
          "param_overrides": [
            {"node_id": "model:base", "params": {"alpha": 1.0}}
          ]
        }
      ]
    }
  ],
  "data_bindings": [
    {
      "node_id": "model:base",
      "input_name": "x",
      "request_id": "data:model.base.x",
      "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
      "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
      "relation_fingerprint": "a3a7e329df35db9f2883a17b8611b7fae6dcaa031875e3ec2c9be1b9e29cbe10",
      "output_representation": "tabular_numeric",
      "feature_set_id": "x",
      "source_ids": ["nir"],
      "require_relations": true
    }
  ],
  "steps": [
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"}
    }
  ],
  "campaign_metadata": {
    "owner": "dsl-test"
  }
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

    assert_eq!(compiled.campaign_template.id, "campaign:dsl.template");
    assert_eq!(compiled.campaign_template.root_seed, Some(123));
    assert_eq!(
        compiled
            .campaign_template
            .split_invocation
            .as_ref()
            .unwrap()
            .id,
        "split:group-kfold"
    );
    assert_eq!(compiled.campaign_template.generation, compiled.generation);
    assert_eq!(
        compiled.data_bindings[&NodeId::new("model:base").unwrap()][0].request_id,
        "data:model.base.x"
    );
    assert_eq!(
        compiled.campaign_template.data_bindings,
        compiled.data_bindings
    );
    assert_eq!(compiled.graph.nodes.len(), 1);
    assert!(compiled
        .graph
        .nodes
        .iter()
        .all(|node| !node.id.as_str().starts_with("split:")));
}

#[test]
fn refuses_data_binding_for_unknown_or_non_data_port() {
    let unknown_input_spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-bad-data-binding",
  "data_bindings": [
    {
      "node_id": "model:base",
      "input_name": "missing",
      "request_id": "data:bad",
      "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
      "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
      "output_representation": "tabular_numeric"
    }
  ],
  "steps": [
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"}
    }
  ]
}"#,
    )
    .unwrap();
    let error = compile_pipeline_dsl_with_generation(&unknown_input_spec).unwrap_err();
    assert!(format!("{error}").contains("unknown input port `missing`"));

    let prediction_input_spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-prediction-port-data-binding",
  "data_bindings": [
    {
      "node_id": "merge:stack.pred_plus_original.meta:ridge",
      "input_name": "b0_oof",
      "request_id": "data:bad.prediction-port",
      "schema_fingerprint": "f97b37872fa22134b508f98fd8e207e5b776b52594fb8f6f5c3e15bee212246b",
      "plan_fingerprint": "7c5431d85574b3f337022fa5d25971d5b5cf445b90331b49938f573ff6901e4d",
      "output_representation": "tabular_numeric"
    }
  ],
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "b0",
          "steps": [
            {
              "kind": "model",
              "id": "branch:b0.model:ridge",
              "operator": {"type": "Ridge"}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge_model",
      "id": "merge:stack.pred_plus_original.meta:ridge",
      "operator": {"type": "RidgeMetaStacker"}
    }
  ]
}"#,
    )
    .unwrap();
    let error = compile_pipeline_dsl_with_generation(&prediction_input_spec).unwrap_err();
    assert!(format!("{error}").contains("targets non-data input"));
}

#[test]
fn extracts_shape_plans_into_compiled_artifact() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-shape-plan-smoke",
  "steps": [
    {
      "kind": "augmentation",
      "id": "augment:synthetic",
      "operator": {"type": "SampleAugmenter"},
      "shape": {
        "input_granularity": "sample",
        "target_granularity": "sample",
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "feature_namespace": "aug.synthetic",
        "augmentation_policy": {
          "sample_scope": "train_only",
          "feature_scope": "none",
          "require_origin_id": true,
          "inherit_group": true,
          "inherit_target": true
        }
      }
    },
    {
      "kind": "transform",
      "id": "transform:select",
      "operator": {"type": "SupervisedFeatureSelector"},
      "shape": {
        "fit_rows": "fold_train",
        "feature_namespace": "selected",
        "selection_policy": {
          "scope": "supervised_fold_train",
          "store_masks": true
        }
      }
    },
    {
      "kind": "model",
      "id": "model:base",
      "operator": {"type": "Ridge"}
    }
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();

    assert_eq!(compiled.shape_plans.len(), 2);
    let augment_plan = compiled
        .shape_plans
        .get(&NodeId::new("augment:synthetic").unwrap())
        .unwrap();
    assert_eq!(
        augment_plan.feature_namespace.as_deref(),
        Some("aug.synthetic")
    );
    assert_eq!(
        augment_plan.augmentation_policy.sample_scope,
        crate::policy::AugmentationScope::TrainOnly
    );
    let select_plan = compiled
        .shape_plans
        .get(&NodeId::new("transform:select").unwrap())
        .unwrap();
    assert_eq!(
        select_plan.selection_policy.scope,
        crate::policy::FeatureSelectionScope::SupervisedFoldTrain
    );
    assert_eq!(compiled.generation.strategy, GenerationStrategy::None);
    compiled.graph.validate().unwrap();
}

#[test]
fn compiles_sequential_filter_and_or_generator_surface() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-generator-or-parity",
  "steps": [
    {
      "kind": "sequential",
      "id": "seq:pre",
      "steps": [
        {
          "kind": "sample_filter",
          "id": "filter:y_outlier",
          "operator": {"class": "nirs4all.operators.filters.YOutlierFilter"},
          "params": {"mode": "any"}
        },
        {
          "kind": "transform",
          "id": "transform:scale",
          "operator": {"class": "sklearn.preprocessing.StandardScaler"}
        }
      ]
    },
    {
      "kind": "generator",
      "id": "generator:model_choices",
      "mode": "or",
      "pick": 1,
      "branches": [
        {
          "id": "pls",
          "steps": [
            {
              "kind": "model",
              "id": "model:pls",
              "operator": {"class": "sklearn.cross_decomposition.PLSRegression"},
              "params": {"n_components": 8}
            }
          ]
        },
        {
          "id": "rf",
          "steps": [
            {
              "kind": "model",
              "id": "model:rf",
              "operator": {"class": "sklearn.ensemble.RandomForestRegressor"},
              "params": {"n_estimators": 64}
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:generated",
      "output_as": "features",
      "include_original_data": false,
      "selectors": [
        {"branch": "generator:model_choices:choice0", "select": "all"}
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    let filter = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "filter:y_outlier")
        .unwrap();
    assert_eq!(filter.kind, NodeKind::Exclude);
    assert_eq!(filter.metadata["dsl_filter_kind"], "sample");

    let generated_models = graph
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Model)
        .collect::<Vec<_>>();
    assert_eq!(generated_models.len(), 2);
    assert!(generated_models
        .iter()
        .all(|node| node.id.as_str().starts_with("gen:generator_model_choices")));
    assert!(generated_models.iter().all(|node| {
        node.metadata
            .get("dsl_generator")
            .and_then(|value| value.as_str())
            == Some("generator:model_choices")
    }));

    let merge_inputs = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "merge:generated")
        .unwrap()
        .ports
        .inputs
        .iter()
        .map(|port| port.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        merge_inputs,
        vec![
            "generator_model_choices_choice0_oof",
            "generator_model_choices_choice1_oof"
        ]
    );
}

#[test]
fn compiles_cartesian_generator_as_explicit_prediction_choices() {
    let spec: PipelineDslSpec = serde_json::from_str(
            r#"{
  "id": "dsl-generator-cartesian-parity",
  "steps": [
    {
      "kind": "generator",
      "id": "generator:cartesian",
      "mode": "cartesian",
      "stages": [
        {
          "id": "preproc",
          "branches": [
            {
              "id": "snv",
              "steps": [
                {
                  "kind": "transform",
                  "id": "transform:snv",
                  "operator": {"class": "nirs4all.operators.transforms.StandardNormalVariate"}
                }
              ]
            },
            {
              "id": "msc",
              "steps": [
                {
                  "kind": "transform",
                  "id": "transform:msc",
                  "operator": {"class": "nirs4all.operators.transforms.MultiplicativeScatterCorrection"}
                }
              ]
            }
          ]
        },
        {
          "id": "model",
          "branches": [
            {
              "id": "ridge",
              "steps": [
                {
                  "kind": "model",
                  "id": "model:ridge",
                  "operator": {"class": "sklearn.linear_model.Ridge"}
                }
              ]
            },
            {
              "id": "lasso",
              "steps": [
                {
                  "kind": "model",
                  "id": "model:lasso",
                  "operator": {"class": "sklearn.linear_model.Lasso"}
                }
              ]
            }
          ]
        }
      ]
    },
    {
      "kind": "merge",
      "id": "merge:cartesian",
      "output_as": "features",
      "include_original_data": false
    }
  ]
}"#,
        )
        .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    let models = graph
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Model)
        .collect::<Vec<_>>();
    assert_eq!(models.len(), 4);
    assert!(models.iter().all(|node| {
        node.metadata
            .get("dsl_generator_mode")
            .and_then(|value| value.as_str())
            == Some("cartesian")
    }));
    let merge = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "merge:cartesian")
        .unwrap();
    assert_eq!(merge.ports.inputs.len(), 4);
    assert_eq!(
        graph
            .edges
            .iter()
            .filter(|edge| edge.target.node_id.as_str() == "merge:cartesian")
            .count(),
        4
    );
}

#[test]
fn refuses_generator_choice_without_prediction_output() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-generator-bad-choice",
  "steps": [
    {
      "kind": "generator",
      "id": "generator:bad",
      "branches": [
        {
          "id": "transform_only",
          "steps": [
            {
              "kind": "transform",
              "id": "transform:only",
              "operator": {"class": "sklearn.preprocessing.StandardScaler"}
            }
          ]
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl(&spec).unwrap_err();
    assert!(format!("{error}").contains("must produce at least one model or merge prediction"));
}

#[test]
fn parses_nirs4all_compat_pipeline_and_fuses_data_generators() {
    let spec = parse_pipeline_dsl_json(
            br#"{
  "id": "dsl-nirs4all-compat-fused",
  "pipeline": [
    {"sources": ["nir"]},
    {"_cartesian_": [
      {"_or_": ["SNV", "MSC", null]},
      {"_or_": [null, {"preprocessing": "SavitzkyGolay", "params": {"window": 11, "deriv": 1}}]}
    ]},
    {"split": {"type": "GroupKFold", "n_splits": 3}},
    {"_chain_": [
      {"_grid_": {"model": ["PLSRegression"], "n_components": [5, 10]}},
      {"_grid_": {"model": ["Ridge"], "alpha": [0.1, 1.0]}},
      {"_sample_": {"model": "SVR", "distribution": "log_uniform", "from": 0.001, "to": 1.0, "num": 2, "tune": ["C", "gamma"], "kernel": "rbf"}}
    ]},
    {"merge": "all"},
    {"model": "Ridge", "id": "model:meta", "params": {"alpha": 0.5}}
  ]
}"#,
        )
        .unwrap();

    assert_eq!(spec.steps.len(), 2);
    assert_eq!(
        spec.split_invocation
            .as_ref()
            .unwrap()
            .params
            .get("type")
            .unwrap(),
        "GroupKFold"
    );

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    let meta = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:meta")
        .unwrap();
    assert_eq!(meta.kind, NodeKind::Model);
    assert!(meta
        .ports
        .inputs
        .iter()
        .any(|port| port.name == "x_original"));
    assert!(graph.edges.iter().any(|edge| {
        edge.target.node_id.as_str() == "model:meta"
            && edge.contract.kind == PortKind::Prediction
            && edge.contract.requires_oof
    }));
    assert!(graph.nodes.iter().any(|node| {
        node.metadata
            .get("dsl_compat_keyword")
            .and_then(serde_json::Value::as_str)
            == Some("preprocessing")
    }));
    assert!(graph.nodes.iter().any(|node| {
        node.kind == NodeKind::Model
            && node.params.contains_key("C")
            && node.params.contains_key("gamma")
    }));
}

#[test]
fn parses_nirs4all_range_attached_to_following_model() {
    let spec = parse_pipeline_dsl_json(
        br#"{
  "id": "dsl-nirs4all-compat-range",
  "pipeline": [
    {"_range_": [5, 15, 5]},
    {"model": "PLSRegression", "id": "model:pls"}
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
    assert_eq!(compiled.generation.dimensions.len(), 1);
    assert_eq!(compiled.generation.dimensions[0].choices.len(), 3);
    assert_eq!(
        compiled.generation.dimensions[0].choices[0].param_overrides[0].params["n_components"],
        5.0
    );
}

#[test]
fn parses_nirs4all_minimal_aliases_plain_classes_and_split_chain() {
    let spec = parse_pipeline_dsl_json(
            br#"{
  "id": "dsl-nirs4all-compat-minimal-aliases",
  "pipeline": [
    "chart_2d",
    {"class": "sklearn.preprocessing.MinMaxScaler", "params": {"feature_range": [0, 1]}},
    {"class": "nirs4all.operators.splitters.SPXYGFold", "params": {"n_splits": 1, "test_size": 0.2}, "group": "Sample_ID"},
    {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 3, "shuffle": true, "random_state": 42}},
    "SNV",
    "PLSRegression"
  ]
}"#,
        )
        .unwrap();

    let split = spec.split_invocation.as_ref().unwrap();
    assert_eq!(split.id, "split:compat.chain");
    let chain = split.params["compat_split_chain"].as_array().unwrap();
    assert_eq!(chain.len(), 2);
    assert_eq!(
        chain[0]["params"]["class"],
        "nirs4all.operators.splitters.SPXYGFold"
    );
    assert_eq!(chain[0]["params"]["group"], "Sample_ID");
    assert_eq!(chain[1]["params"]["class"], "sklearn.model_selection.KFold");

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    assert!(graph.nodes.iter().any(|node| node.kind == NodeKind::Chart));
    assert!(graph.nodes.iter().any(|node| {
        node.kind == NodeKind::Transform
            && node.operator.as_ref().unwrap()["class"] == "sklearn.preprocessing.MinMaxScaler"
    }));
    assert!(graph.nodes.iter().any(|node| {
        node.kind == NodeKind::Transform && node.operator.as_ref().unwrap().as_str() == Some("SNV")
    }));
    assert!(graph.nodes.iter().any(|node| {
        node.kind == NodeKind::Model
            && node.operator.as_ref().unwrap().as_str() == Some("PLSRegression")
    }));
}

#[test]
fn registry_reclassifies_non_heuristic_minimal_aliases_before_compile() {
    let spec = parse_pipeline_dsl_json(
        br#"{
  "id": "dsl-registry-minimal-aliases",
  "pipeline": [
    "SNV",
    "ElasticSpectra"
  ]
}"#,
    )
    .unwrap();
    let mut registry = ControllerRegistry::new();
    registry
        .register(registry_manifest(
            "controller:transformer.mixin",
            NodeKind::Transform,
            &["SNV"],
        ))
        .unwrap();
    registry
        .register(registry_manifest(
            "controller:elastic.spectra",
            NodeKind::Model,
            &["ElasticSpectra"],
        ))
        .unwrap();

    let compiled =
        compile_pipeline_dsl_with_generation_and_controller_registry(&spec, &registry).unwrap();
    let model = compiled
        .graph
        .nodes
        .iter()
        .find(|node| {
            node.operator.as_ref().and_then(serde_json::Value::as_str) == Some("ElasticSpectra")
        })
        .unwrap();

    assert_eq!(model.kind, NodeKind::Model);
    assert_eq!(model.metadata[DSL_REGISTRY_INFERRED_KIND], "model");
    assert_eq!(model.metadata[DSL_COMPAT_ORIGINAL_KEYWORD], "preprocessing");
    assert!(compiled.graph.nodes.iter().any(|node| {
        node.kind == NodeKind::Transform
            && node.operator.as_ref().and_then(serde_json::Value::as_str) == Some("SNV")
    }));
}

#[test]
fn parses_nirs4all_named_step_wrapper_and_plain_class_model() {
    let spec = parse_pipeline_dsl_json(
            br#"{
  "id": "dsl-nirs4all-compat-named-step",
  "pipeline": [
    {"name": "scaled", "step": {"class": "sklearn.preprocessing.StandardScaler"}},
    {"class": "sklearn.ensemble.RandomForestRegressor", "params": {"n_estimators": 10, "random_state": 42}}
  ]
}"#,
        )
        .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    let scaled = graph
        .nodes
        .iter()
        .find(|node| node.kind == NodeKind::Transform)
        .unwrap();
    assert_eq!(scaled.metadata["dsl_name"], "scaled");
    let model = graph
        .nodes
        .iter()
        .find(|node| node.kind == NodeKind::Model)
        .unwrap();
    assert_eq!(
        model.operator.as_ref().unwrap()["class"],
        "sklearn.ensemble.RandomForestRegressor"
    );
    assert_eq!(model.params["n_estimators"], 10);
}

#[test]
fn compiles_tuner_as_external_prediction_node() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-tuner",
  "steps": [
    {
      "kind": "tuner",
      "id": "tuner:optuna",
      "operator": "OptunaTuner",
      "params": {"sampler": "tpe"},
      "tuning": {"n_trials": 4, "metric": "rmse"}
    },
    {
      "kind": "merge_model",
      "id": "model:meta",
      "operator": "Ridge"
    }
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    let tuner = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "tuner:optuna")
        .unwrap();
    assert_eq!(tuner.kind, NodeKind::Tuner);
    assert_eq!(
        tuner.operator.as_ref().unwrap().as_str(),
        Some("OptunaTuner")
    );
    assert_eq!(tuner.metadata["dsl_tuning"]["n_trials"], 4);
    assert!(graph.edges.iter().any(|edge| {
        edge.source.node_id.as_str() == "tuner:optuna"
            && edge.source.port_name == "oof"
            && edge.target.node_id.as_str() == "model:meta"
            && edge.contract.kind == PortKind::Prediction
            && edge.contract.requires_oof
            && edge.contract.requires_fold_alignment
    }));
}

#[test]
fn parses_compat_tuner_minimal_alias_and_wrappers() {
    let spec = parse_pipeline_dsl_json(
        br#"{
  "id": "dsl-compat-tuner",
  "pipeline": [
    "SNV",
    {"tuner": "OptunaTuner", "id": "tuner:compat", "n_trials": 3, "metric": "rmse"},
    {"merge": "all"},
    {"model": "Ridge"}
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    let transform = graph
        .nodes
        .iter()
        .find(|node| node.kind == NodeKind::Transform)
        .unwrap();
    assert_eq!(transform.operator.as_ref().unwrap().as_str(), Some("SNV"));
    let tuner = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "tuner:compat")
        .unwrap();
    assert_eq!(tuner.kind, NodeKind::Tuner);
    assert_eq!(tuner.params["n_trials"], 3);
    assert_eq!(tuner.metadata["dsl_compat_keyword"], "tuner");
}

#[test]
fn parses_bare_tuner_alias_as_tuner_node() {
    let spec = parse_pipeline_dsl_json(
        br#"{
  "id": "dsl-bare-tuner-alias",
  "pipeline": ["SNV", "OptunaTuner"]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    assert!(graph.nodes.iter().any(|node| {
        node.kind == NodeKind::Transform && node.operator.as_ref().unwrap().as_str() == Some("SNV")
    }));
    assert!(graph.nodes.iter().any(|node| {
        node.kind == NodeKind::Tuner
            && node.operator.as_ref().unwrap().as_str() == Some("OptunaTuner")
    }));
}

#[test]
fn compiles_runtime_data_generation_as_external_generator_node() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-runtime-data-generation",
  "steps": [
    {
      "kind": "generation",
      "id": "generator:synthetic.train",
      "operator": "SMOTE",
      "params": {"ratio": 0.5},
      "shape": {
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "augmentation_policy": {
          "sample_scope": "train_only",
          "feature_scope": "none",
          "require_origin_id": true,
          "inherit_group": true,
          "inherit_target": true
        }
      }
    },
    {
      "kind": "model",
      "id": "model:ridge",
      "operator": "Ridge"
    }
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
    compiled.graph.validate().unwrap();
    let generator = compiled
        .graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "generator:synthetic.train")
        .unwrap();
    assert_eq!(generator.kind, NodeKind::Generator);
    assert_eq!(generator.operator.as_ref().unwrap().as_str(), Some("SMOTE"));
    assert_eq!(generator.metadata["dsl_generation_kind"], "data");
    assert!(compiled
        .shape_plans
        .contains_key(&NodeId::new("generator:synthetic.train").unwrap()));
    assert!(compiled.graph.edges.iter().any(|edge| {
        edge.source.node_id.as_str() == "generator:synthetic.train"
            && edge.source.port_name == "x_out"
            && edge.target.node_id.as_str() == "model:ridge"
            && edge.target.port_name == "x"
            && edge.contract.kind == PortKind::Data
    }));
}

#[test]
fn parses_compat_runtime_generation_step() {
    let spec = parse_pipeline_dsl_json(
        br#"{
  "id": "dsl-compat-runtime-generation",
  "pipeline": [
    {
      "generation": "SMOTE",
      "id": "generator:compat.synthetic",
      "generation_params": {"ratio": 0.25},
      "shape": {
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "augmentation_policy": {
          "sample_scope": "train_only",
          "feature_scope": "none",
          "require_origin_id": true,
          "inherit_group": true,
          "inherit_target": true
        }
      }
    },
    "Ridge"
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
    let generator = compiled
        .graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "generator:compat.synthetic")
        .unwrap();
    assert_eq!(generator.kind, NodeKind::Generator);
    assert_eq!(generator.params["ratio"], 0.25);
    assert_eq!(generator.metadata["dsl_compat_keyword"], "data_generation");
}

#[test]
fn parses_nirs4all_compat_feature_branch_merge_dict() {
    let spec = parse_pipeline_dsl_json(
        br#"{
  "id": "dsl-nirs4all-compat-feature-merge",
  "pipeline": [
    {
      "branch": {
        "snv": ["SNV"],
        "msc": ["MSC"]
      }
    },
    {
      "merge": {
        "features": "all",
        "output_as": "features",
        "on_missing": "error"
      }
    },
    "PLSRegression"
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    graph.validate().unwrap();
    let merge = graph
        .nodes
        .iter()
        .find(|node| node.kind == NodeKind::FeatureJoin)
        .unwrap();
    assert_eq!(merge.metadata["merge_mode"], "features");
    assert_eq!(merge.metadata["on_missing"], "error");
    assert!(merge.metadata.contains_key("dsl_compat_merge"));
    assert!(merge.ports.inputs.iter().any(|port| port.name == "snv_x"));
    assert!(merge.ports.inputs.iter().any(|port| port.name == "msc_x"));
    assert!(graph.nodes.iter().any(|node| node.kind == NodeKind::Model
        && node.operator.as_ref().unwrap().as_str() == Some("PLSRegression")));
}

#[test]
fn published_pipeline_dsl_schema_declares_current_contract() {
    let schema: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../docs/contracts/pipeline_dsl.schema.json"
    ))
    .unwrap();

    assert_eq!(schema["$id"], PIPELINE_DSL_SCHEMA_ID);
    assert!(schema["oneOf"].is_array());
    assert!(schema["$defs"]["canonical_step_kind"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str() == Some("generator")));
    assert!(schema["$defs"]["canonical_step_kind"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str() == Some("data_generation")));
    assert!(schema["$defs"]["canonical_step_kind"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str() == Some("tuner")));
    assert!(schema["$defs"]["compat_generator_key"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str() == Some("_cartesian_")));
    assert!(schema["$defs"]["compat_step_object"]["properties"]
        .as_object()
        .unwrap()
        .contains_key("class"));
    assert!(schema["$defs"]["compat_step_object"]["properties"]
        .as_object()
        .unwrap()
        .contains_key("step"));
    assert!(schema["$defs"]["pipeline_unit_contract"]["properties"]
        .as_object()
        .unwrap()
        .contains_key("unit_level"));
    assert!(schema["$defs"]["entity_unit_level"]["enum"]
        .as_array()
        .unwrap()
        .iter()
        .any(|value| value.as_str() == Some("observation")));
}

#[test]
fn refuses_unsafe_shape_plan_from_dsl() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-unsafe-shape-plan",
  "steps": [
    {
      "kind": "augmentation",
      "id": "augment:bad",
      "operator": {"type": "LeakyAugmenter"},
      "shape": {
        "augmentation_policy": {
          "sample_scope": "all_partitions"
        }
      }
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
    assert!(format!("{error}").contains("sample augmentation over all partitions"));
}

#[test]
fn refuses_augmentation_without_shape_plan() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-augmentation-without-shape",
  "steps": [
    {
      "kind": "augmentation",
      "id": "augment:missing-shape",
      "operator": {"type": "GaussianNoise"}
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
    assert!(format!("{error}").contains("requires a shape plan"));
}

#[test]
fn refuses_data_generation_without_shape_plan() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-generation-without-shape",
  "steps": [
    {
      "kind": "data_generation",
      "id": "generator:missing-shape",
      "operator": {"type": "SMOTE"}
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl_with_generation(&spec).unwrap_err();
    assert!(format!("{error}").contains("requires a shape plan"));
}

#[test]
fn refuses_branch_without_prediction_or_data_output() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-bad-branch",
  "steps": [
    {
      "kind": "branch",
      "branches": [
        {
          "id": "b0",
          "steps": [
            {
              "kind": "y_transform",
              "id": "target:only",
              "operator": {"type": "StandardScaler"}
            }
          ]
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();

    let error = compile_pipeline_dsl(&spec).unwrap_err();
    assert!(format!("{error}")
        .contains("must produce at least one model, merge prediction or transformed data"));
}

#[test]
fn dsl_top_level_inner_cv_maps_to_campaign_template() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-inner-cv-campaign",
  "inner_cv": {"kind": "kfold", "n_splits": 4, "shuffle": true, "seed": 7},
  "steps": [
    {"kind": "model", "id": "model:base", "operator": {"type": "Ridge"}, "params": {"alpha": 0.5}}
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
    match compiled.campaign_template.inner_cv {
        Some(crate::fold::NestedCvSpec::KFold(ref k)) => {
            assert_eq!(k.n_splits, 4);
            assert!(k.shuffle);
            assert_eq!(k.seed, Some(7));
        }
        ref other => panic!("expected campaign-level KFold inner_cv, got {other:?}"),
    }
}

#[test]
fn dsl_model_step_inner_cv_maps_to_node_metadata() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-inner-cv-node",
  "steps": [
    {
      "kind": "model",
      "id": "model:meta",
      "operator": {"type": "Ridge"},
      "inner_cv": {"kind": "group_kfold", "n_splits": 3}
    }
  ]
}"#,
    )
    .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    let node = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:meta")
        .expect("compiled model node exists");
    let value = node
        .metadata
        .get("dsl_inner_cv")
        .expect("node carries dsl_inner_cv metadata");
    let inner: crate::fold::NestedCvSpec = serde_json::from_value(value.clone()).unwrap();
    match inner {
        crate::fold::NestedCvSpec::GroupKFold(ref g) => assert_eq!(g.n_splits, 3),
        other => panic!("expected node-local GroupKFold inner_cv, got {other:?}"),
    }
}

#[test]
fn dsl_absent_inner_cv_leaves_campaign_and_nodes_unset() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-no-inner-cv",
  "steps": [
    {"kind": "model", "id": "model:base", "operator": {"type": "Ridge"}}
  ]
}"#,
    )
    .unwrap();

    let compiled = compile_pipeline_dsl_with_generation(&spec).unwrap();
    assert!(compiled.campaign_template.inner_cv.is_none());
    for node in &compiled.graph.nodes {
        assert!(!node.metadata.contains_key("dsl_inner_cv"));
    }
}

#[test]
fn compat_pipeline_preserves_campaign_and_model_inner_cv() {
    // nirs4all-compatible dict form ("pipeline" key) routes through the compat
    // lowerer; campaign-global and node-local inner_cv must survive lowering.
    let spec = parse_pipeline_dsl_json(
        br#"{
  "id": "dsl-compat-inner-cv",
  "inner_cv": {"kind": "kfold", "n_splits": 5, "shuffle": false, "seed": 3},
  "pipeline": [
    {"split": {"type": "KFold", "n_splits": 4}},
    {"model": "Ridge", "id": "model:base", "inner_cv": {"kind": "group_kfold", "n_splits": 3}}
  ]
}"#,
    )
    .unwrap();

    match spec.inner_cv {
        Some(crate::fold::NestedCvSpec::KFold(ref k)) => assert_eq!(k.n_splits, 5),
        ref other => panic!("expected compat campaign-global KFold inner_cv, got {other:?}"),
    }

    let graph = compile_pipeline_dsl(&spec).unwrap();
    let node = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:base")
        .expect("compat model node exists");
    let inner: crate::fold::NestedCvSpec =
        serde_json::from_value(node.metadata.get("dsl_inner_cv").cloned().unwrap()).unwrap();
    match inner {
        crate::fold::NestedCvSpec::GroupKFold(ref g) => assert_eq!(g.n_splits, 3),
        other => panic!("expected compat node-local GroupKFold inner_cv, got {other:?}"),
    }
}

#[test]
fn compat_merge_model_collapse_preserves_inner_cv() {
    // The compat `merge` + `model` stacker shorthand collapses into a
    // merge-model step; its node-local inner_cv must reach the graph node.
    let spec = parse_pipeline_dsl_json(
            br#"{
  "id": "dsl-compat-merge-inner-cv",
  "pipeline": [
    {"_chain_": [
      {"_grid_": {"model": ["PLSRegression"], "n_components": [5, 10]}},
      {"_grid_": {"model": ["Ridge"], "alpha": [0.1, 1.0]}}
    ]},
    {"merge": "predictions"},
    {"model": "Ridge", "id": "model:meta", "params": {"alpha": 0.5}, "inner_cv": {"kind": "kfold", "n_splits": 4, "shuffle": false, "seed": null}}
  ]
}"#,
        )
        .unwrap();

    let graph = compile_pipeline_dsl(&spec).unwrap();
    let node = graph
        .nodes
        .iter()
        .find(|node| node.id.as_str() == "model:meta")
        .expect("compat merge-model node exists");
    let inner: crate::fold::NestedCvSpec =
        serde_json::from_value(node.metadata.get("dsl_inner_cv").cloned().unwrap()).unwrap();
    match inner {
        crate::fold::NestedCvSpec::KFold(ref k) => assert_eq!(k.n_splits, 4),
        other => panic!("expected merge-model KFold inner_cv, got {other:?}"),
    }
}

// ----- plan-time data-aware branch fan-out (Slice 2 keystone) -----

const FANOUT_FP: &str = "1111111111111111111111111111111111111111111111111111111111111111";

/// Build an envelope whose coordinator relations carry the given
/// `(metadata_value, tags)` per sample so a by_metadata/by_tag fan-out has
/// distinct values to discover.
fn fanout_envelope(rows: &[(&str, &str, &[&str])]) -> crate::data::ExternalDataPlanEnvelope {
    use crate::ids::{ObservationId, SampleId};
    use crate::relation::{SampleRelation, SampleRelationSet};
    let records = rows
        .iter()
        .map(|(sample, site, tags)| {
            let mut relation = SampleRelation::new(
                ObservationId::new(format!("obs:{sample}")).unwrap(),
                SampleId::new(format!("sample:{sample}")).unwrap(),
            );
            relation
                .metadata
                .insert("site".to_string(), serde_json::json!(site));
            relation.tags = tags.iter().map(|tag| (*tag).to_string()).collect();
            relation
        })
        .collect::<Vec<_>>();
    let relations = SampleRelationSet { records };
    relations.validate().unwrap();
    crate::data::ExternalDataPlanEnvelope {
        schema_version: crate::data::EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION,
        schema_fingerprint: FANOUT_FP.to_string(),
        plan_fingerprint: FANOUT_FP.to_string(),
        relation_fingerprint: Some(relations.fingerprint().unwrap()),
        coordinator_relations: Some(relations),
    }
}

fn auto_separation_by_metadata_spec() -> PipelineDslSpec {
    serde_json::from_str(
            r#"{
  "id": "dsl-fanout-by-metadata",
  "steps": [
    {
      "kind": "branch",
      "mode": "by_metadata",
      "selector": {"metadata_key": "site"},
      "metadata": {"auto_separate": true},
      "branches": [
        {
          "id": "per_site",
          "steps": [
            {"kind": "transform", "id": "transform:snv", "operator": {"type": "StandardNormalVariate"}},
            {"kind": "model", "id": "model:site", "operator": {"type": "PLSRegression"}}
          ]
        }
      ]
    }
  ]
}"#,
        )
        .unwrap()
}

#[test]
fn fans_out_by_metadata_into_one_branch_per_sorted_value() {
    let spec = auto_separation_by_metadata_spec();
    // Insertion order C, A, B — fan-out must sort to A, B, C.
    let envelope = fanout_envelope(&[
        ("s1", "C", &[]),
        ("s2", "A", &[]),
        ("s3", "B", &[]),
        ("s4", "A", &[]),
    ]);

    let expanded = fan_out_data_aware_branches(&spec, &envelope).unwrap();

    let PipelineDslStep::Branch(branch_step) = &expanded.steps[0] else {
        panic!("expected a branch step");
    };
    // Three distinct sites -> three branches, sorted A,B,C.
    let ids: Vec<&str> = branch_step.branches.iter().map(|b| b.id.as_str()).collect();
    assert_eq!(ids, vec!["per_site__A", "per_site__B", "per_site__C"]);
    // The auto_separate marker is consumed so a re-run is a no-op.
    assert!(!branch_step.metadata.contains_key("auto_separate"));
    // Each branch targets exactly its own metadata value.
    assert_eq!(
        branch_step.branches[0].selector.as_ref().unwrap()["metadata"]["site"],
        "A"
    );
    assert_eq!(
        branch_step.branches[2].selector.as_ref().unwrap()["metadata"]["site"],
        "C"
    );

    // The expanded spec compiles into N concrete branch model nodes, one per
    // partition, each scoped by its own BranchViewPlan — RETAIN ALL.
    let compiled = compile_pipeline_dsl_with_generation(&expanded).unwrap();
    assert_eq!(compiled.branch_view_plans.len(), 3);
    let mut sites: Vec<String> = compiled
        .branch_view_plans
        .iter()
        .map(|plan| plan.selector.metadata["site"].as_str().unwrap().to_string())
        .collect();
    sites.sort();
    assert_eq!(sites, vec!["A", "B", "C"]);
    // One model node per partition, each carrying its own branch view.
    for site in ["A", "B", "C"] {
        let node = compiled
            .graph
            .nodes
            .iter()
            .find(|node| node.id.as_str() == format!("model:site__{site}"))
            .unwrap_or_else(|| panic!("missing per-partition model node for site {site}"));
        assert_eq!(
            node.metadata["dsl_branch_view_plan"]["selector"]["metadata"]["site"],
            site
        );
    }
    compiled.graph.validate().unwrap();
}

#[test]
fn fans_out_by_tag_into_one_branch_per_sorted_tag() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-fanout-by-tag",
  "steps": [
    {
      "kind": "branch",
      "mode": "by_tag",
      "metadata": {"auto_separate": true},
      "branches": [
        {
          "id": "per_tag",
          "steps": [
            {"kind": "model", "id": "model:tag", "operator": {"type": "Ridge"}}
          ]
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let envelope = fanout_envelope(&[
        ("s1", "x", &["red", "blue"]),
        ("s2", "y", &["blue"]),
        ("s3", "z", &["green"]),
    ]);

    let expanded = fan_out_data_aware_branches(&spec, &envelope).unwrap();
    let PipelineDslStep::Branch(branch_step) = &expanded.steps[0] else {
        panic!("expected a branch step");
    };
    let ids: Vec<&str> = branch_step.branches.iter().map(|b| b.id.as_str()).collect();
    assert_eq!(ids, vec!["per_tag__blue", "per_tag__green", "per_tag__red"]);
    assert_eq!(
        branch_step.branches[0].selector.as_ref().unwrap()["tags"][0],
        "blue"
    );

    let compiled = compile_pipeline_dsl_with_generation(&expanded).unwrap();
    assert_eq!(compiled.branch_view_plans.len(), 3);
    assert_eq!(compiled.branch_view_plans[0].mode, BranchViewMode::ByTag);
}

#[test]
fn fan_out_is_deterministic_byte_identical() {
    let spec = auto_separation_by_metadata_spec();
    let envelope = fanout_envelope(&[("s1", "B", &[]), ("s2", "A", &[])]);

    let first = fan_out_data_aware_branches(&spec, &envelope).unwrap();
    let second = fan_out_data_aware_branches(&spec, &envelope).unwrap();
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap(),
        "identical data must expand to a byte-identical spec"
    );
    // The fan-out fingerprint is recorded and stable across runs.
    let fp = first.metadata[DSL_DATA_AWARE_FANOUT_METADATA_KEY]["fingerprint"].clone();
    assert_eq!(
        fp,
        second.metadata[DSL_DATA_AWARE_FANOUT_METADATA_KEY]["fingerprint"]
    );
    assert!(fp.as_str().unwrap().len() == 64);
}

#[test]
fn leaves_explicit_and_unmarked_branches_untouched() {
    // Same shape as the auto-separation spec but WITHOUT the auto_separate
    // marker -> must be left exactly as written (compiles as a plain
    // single-branch explicit step would, here it errors for lacking a
    // selector, proving fan-out did not touch it).
    let mut spec = auto_separation_by_metadata_spec();
    if let PipelineDslStep::Branch(step) = &mut spec.steps[0] {
        step.metadata.remove("auto_separate");
    }
    let envelope = fanout_envelope(&[("s1", "A", &[]), ("s2", "B", &[])]);
    let expanded = fan_out_data_aware_branches(&spec, &envelope).unwrap();
    // Unchanged: still one branch, no fan-out metadata recorded.
    let PipelineDslStep::Branch(branch_step) = &expanded.steps[0] else {
        panic!("expected a branch step");
    };
    assert_eq!(branch_step.branches.len(), 1);
    assert_eq!(branch_step.branches[0].id, "per_site");
    assert!(!expanded
        .metadata
        .contains_key(DSL_DATA_AWARE_FANOUT_METADATA_KEY));
}

#[test]
fn fan_out_requires_relations_in_envelope() {
    let spec = auto_separation_by_metadata_spec();
    let envelope = crate::data::ExternalDataPlanEnvelope {
        schema_version: crate::data::EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION,
        schema_fingerprint: FANOUT_FP.to_string(),
        plan_fingerprint: FANOUT_FP.to_string(),
        relation_fingerprint: None,
        coordinator_relations: None,
    };
    let error = fan_out_data_aware_branches(&spec, &envelope)
        .unwrap_err()
        .to_string();
    assert!(error.contains("requires coordinator relations"), "{error}");
}

#[test]
fn fan_out_errors_when_no_partition_values_discovered() {
    let spec = auto_separation_by_metadata_spec();
    // Relations carry NO "site" metadata key -> nothing to discover.
    use crate::ids::{ObservationId, SampleId};
    use crate::relation::{SampleRelation, SampleRelationSet};
    let relations = SampleRelationSet {
        records: vec![SampleRelation::new(
            ObservationId::new("obs:s1").unwrap(),
            SampleId::new("sample:s1").unwrap(),
        )],
    };
    let envelope = crate::data::ExternalDataPlanEnvelope {
        schema_version: crate::data::EXTERNAL_DATA_PLAN_ENVELOPE_SCHEMA_VERSION,
        schema_fingerprint: FANOUT_FP.to_string(),
        plan_fingerprint: FANOUT_FP.to_string(),
        relation_fingerprint: Some(relations.fingerprint().unwrap()),
        coordinator_relations: Some(relations),
    };
    let error = fan_out_data_aware_branches(&spec, &envelope)
        .unwrap_err()
        .to_string();
    assert!(error.contains("discovered no partition values"), "{error}");
}

#[test]
fn fan_out_clones_top_level_data_bindings_per_branch() {
    // The executable DSL (from run_backend) carries top-level data_bindings
    // keyed by node id. Fan-out must clone+rewrite them per discovered
    // partition so each per-branch model node has its binding (no dangle, no
    // collision), and the original template-id binding must be gone.
    let mut spec = auto_separation_by_metadata_spec();
    spec.data_bindings = vec![crate::data::DataBinding {
        node_id: NodeId::new("model:site").unwrap(),
        input_name: "x".to_string(),
        request_id: "req".to_string(),
        schema_fingerprint: FANOUT_FP.to_string(),
        plan_fingerprint: FANOUT_FP.to_string(),
        relation_fingerprint: Some(FANOUT_FP.to_string()),
        output_representation: "tabular_numeric".to_string(),
        feature_set_id: Some("x".to_string()),
        source_ids: Vec::new(),
        require_relations: false,
        view_policy: Default::default(),
        metadata: BTreeMap::new(),
    }];
    let envelope = fanout_envelope(&[("s1", "A", &[]), ("s2", "B", &[])]);

    let expanded = fan_out_data_aware_branches(&spec, &envelope).unwrap();

    let bound_nodes: Vec<&str> = expanded
        .data_bindings
        .iter()
        .map(|binding| binding.node_id.as_str())
        .collect();
    assert_eq!(bound_nodes, vec!["model:site__A", "model:site__B"]);
    // The dangling original-id binding is gone.
    assert!(!expanded
        .data_bindings
        .iter()
        .any(|binding| binding.node_id.as_str() == "model:site"));
    // The rewritten bindings still target real per-branch nodes after compile.
    let compiled = compile_pipeline_dsl_with_generation(&expanded).unwrap();
    for site in ["A", "B"] {
        let node_id = format!("model:site__{site}");
        assert!(
            compiled
                .data_bindings
                .contains_key(&NodeId::new(&node_id).unwrap()),
            "binding missing for fanned node {node_id}"
        );
    }
    compiled.graph.validate().unwrap();
}

#[test]
fn fan_out_rejects_merge_inside_auto_separation_template() {
    let spec: PipelineDslSpec = serde_json::from_str(
        r#"{
  "id": "dsl-fanout-merge-template",
  "steps": [
    {
      "kind": "branch",
      "mode": "by_metadata",
      "selector": {"metadata_key": "site"},
      "metadata": {"auto_separate": true},
      "branches": [
        {
          "id": "per_site",
          "steps": [
            {"kind": "model", "id": "model:a", "operator": {"type": "Ridge"}},
            {"kind": "merge_model", "id": "model:meta", "operator": {"type": "Ridge"}}
          ]
        }
      ]
    }
  ]
}"#,
    )
    .unwrap();
    let envelope = fanout_envelope(&[("s1", "A", &[]), ("s2", "B", &[])]);
    let error = fan_out_data_aware_branches(&spec, &envelope)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("does not support a `merge_model` step"),
        "{error}"
    );
}

#[test]
fn fan_out_rejects_generation_override_on_fanned_node() {
    let mut spec = auto_separation_by_metadata_spec();
    spec.generation_dimensions = vec![PipelineDslGenerationDimension {
        name: "dim".to_string(),
        choices: vec![PipelineDslGenerationChoice {
            label: "c0".to_string(),
            value: None,
            param_overrides: vec![PipelineDslGenerationParamOverride {
                // Targets the template model node that fan-out multiplies.
                node_id: NodeId::new("model:site").unwrap(),
                params: BTreeMap::from([("alpha".to_string(), serde_json::json!(0.1))]),
            }],
        }],
    }];
    let envelope = fanout_envelope(&[("s1", "A", &[]), ("s2", "B", &[])]);
    let error = fan_out_data_aware_branches(&spec, &envelope)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("generation param_override targeting node `model:site`"),
        "{error}"
    );
}
