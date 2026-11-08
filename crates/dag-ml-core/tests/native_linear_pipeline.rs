//! Portable API coverage; deliberately has no `methods-optimizer` gate or n4m import.
//! Intended checks (NOT executed by the source author):
//! `cargo test -p dag-ml-core --no-default-features --test native_linear_pipeline`
//! `cargo check -p dag-ml-core --no-default-features --target wasm32-unknown-unknown`
//! IO: check/test `nirs4all-io-dagml --lib --no-default-features`, then the default CLI.

use dag_ml_core::{
    methods_native_controller_ids, n4m_host_controller_specs, native_pipeline_training_request,
    ControllerManifest, DataBinding, ExternalDataPlanEnvelope, GroupId, N4mRole,
    NativeLinearPipelineFacts, NativePipelineRecipe, SampleId, SampleRelationSet,
    TrainingDataIdentity, TrainingRequest, METHODS_ESTIMATOR_EXECUTABLE_ROLES,
};
use serde_json::{json, Value};

fn recipe() -> NativePipelineRecipe {
    serde_json::from_value(json!({
        "steps": [
            {"method_id":"preprocessing.scaling.standard_scale","role":"transformer","params":{}},
            {"method_id":"models.regularized.ridge","role":"regressor","params":{"scale_x":false}}
        ],
        "candidates": [{"alpha":0.1},{"alpha":1.0}]
    }))
    .unwrap()
}

fn manifests() -> Vec<ControllerManifest> {
    // The unused classifier must also survive in the signed manifest closure.
    n4m_host_controller_specs(&json!({
        "abi":"2.20",
        "methods":[
            {"method_id":"preprocessing.scaling.standard_scale","roles":["transformer"],"node_kinds":["transform"]},
            {"method_id":"models.regularized.ridge","roles":["regressor"],"node_kinds":["model"]},
            {"method_id":"fixture.classifier","roles":["classifier"],"node_kinds":["model"]}
        ]
    }).to_string()).unwrap().iter().map(|s| s.derive().unwrap()).collect()
}

fn facts() -> NativeLinearPipelineFacts {
    // Same recipe/target order and unsorted identity shape as Core's retained
    // fixture, with synthetic content fingerprints: IO extraction is not tested here.
    let relations: SampleRelationSet = serde_json::from_value(json!({"records": [
        {"observation_id":"o6","sample_id":"train-06","tags":["partition:train"]},
        {"observation_id":"o5","sample_id":"train-05","tags":["partition:train"]},
        {"observation_id":"o4","sample_id":"train-04","tags":["partition:train"]},
        {"observation_id":"o3","sample_id":"train-03","tags":["partition:train"]},
        {"observation_id":"o2","sample_id":"train-02","tags":["partition:train"]},
        {"observation_id":"o1","sample_id":"train-01","tags":["partition:train"]}
    ]}))
    .unwrap();
    let envelope = ExternalDataPlanEnvelope {
        schema_version: 1,
        schema_fingerprint: "1".repeat(64),
        plan_fingerprint: "2".repeat(64),
        relation_fingerprint: Some(relations.fingerprint().unwrap()),
        data_content_fingerprint: Some("3".repeat(64)),
        target_content_fingerprint: Some("4".repeat(64)),
        coordinator_relations: Some(relations.clone()),
        predict_cohort: None,
    };
    NativeLinearPipelineFacts {
        envelope,
        relations,
        feature_set_id: "spectra".into(),
        source_id: "spectra".into(),
        target_names: vec!["protein".into(), "moisture".into()],
        input_sample_ids: Some(
            [
                "train-04", "train-01", "train-06", "train-03", "train-05", "train-02",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        ),
        declared_folds: vec![
            (vec![1, 3, 5], vec![0, 2, 4]),
            (vec![0, 2, 4], vec![1, 3, 5]),
        ],
    }
}

fn refresh_relations(facts: &mut NativeLinearPipelineFacts) {
    facts.envelope.relation_fingerprint = Some(facts.relations.fingerprint().unwrap());
    facts.envelope.coordinator_relations = Some(facts.relations.clone());
}

// Frozen request fields transcribed from the exact pre-move Core Raw base and
// standard_scale/Ridge recipe. Only opaque envelope/catalog identities are
// filled below. This is a concrete fixture, not a second request compiler.
const LEGACY_REQUEST: &str = r#"
{
  "schema_version": 1,
  "request_id": "training:core:canonical-raw-pls",
  "plan_id": "plan:core:canonical-raw-pls",
  "graph": {
    "id": "core-canonical-raw-pls",
    "interface": {
      "inputs": [
        {
          "name": "x",
          "kind": "data",
          "representation": "tabular_numeric",
          "cardinality": "one",
          "description": "selected IO numeric source"
        }
      ],
      "outputs": [
        {
          "name": "prediction",
          "kind": "prediction",
          "representation": null,
          "cardinality": "one",
          "description": "PLS prediction"
        }
      ]
    },
    "nodes": [
      {
        "id": "transform:0",
        "kind": "transform",
        "operator": "n4m:preprocessing.scaling.standard_scale",
        "params": {
          "method_id": "preprocessing.scaling.standard_scale"
        },
        "ports": {
          "inputs": [
            {
              "name": "x",
              "kind": "data",
              "representation": "tabular_numeric",
              "cardinality": "one",
              "description": ""
            }
          ],
          "outputs": [
            {
              "name": "x_out",
              "kind": "data",
              "representation": "tabular_numeric",
              "cardinality": "one",
              "description": ""
            }
          ]
        },
        "metadata": {},
        "seed_label": null
      },
      {
        "id": "model:pls",
        "kind": "model",
        "operator": "n4m:models.regularized.ridge",
        "params": {
          "scale_x": false,
          "method_id": "models.regularized.ridge"
        },
        "ports": {
          "inputs": [
            {
              "name": "x",
              "kind": "data",
              "representation": "tabular_numeric",
              "cardinality": "one",
              "description": ""
            }
          ],
          "outputs": [
            {
              "name": "oof",
              "kind": "prediction",
              "representation": null,
              "cardinality": "one",
              "description": ""
            }
          ]
        },
        "metadata": {},
        "seed_label": null
      }
    ],
    "edges": [
      {
        "source": {
          "node_id": "transform:0",
          "port_name": "x_out"
        },
        "target": {
          "node_id": "model:pls",
          "port_name": "x"
        },
        "contract": {
          "kind": "data",
          "representation": "tabular_numeric",
          "requires_oof": false,
          "requires_fold_alignment": true,
          "propagates_lineage": true
        }
      }
    ],
    "search_space_fingerprint": null,
    "metadata": {}
  },
  "campaign": {
    "id": "campaign:core:canonical-raw-pls",
    "root_seed": 91,
    "leakage_policy": {
      "split_unit": "sample",
      "forbid_origin_cross_fold": true,
      "allow_observation_split_with_shared_target": false,
      "require_group_ids": false,
      "unsafe_flags": []
    },
    "aggregation_policy": {
      "aggregation_level": "sample",
      "method": "mean",
      "weights": "none",
      "emit_parallel_metrics": true,
      "selection_metric_level": "sample",
      "store_raw_predictions": true,
      "store_aggregated_predictions": true
    },
    "split_invocation": {
      "id": "native:declared-folds",
      "controller_id": null,
      "leakage_policy": {
        "split_unit": "sample",
        "forbid_origin_cross_fold": true,
        "allow_observation_split_with_shared_target": false,
        "require_group_ids": false,
        "unsafe_flags": []
      },
      "params": {
        "kind": "precomputed"
      },
      "fold_set": {
        "id": "native:declared-folds",
        "sample_ids": [
          "train-04",
          "train-01",
          "train-06",
          "train-03",
          "train-05",
          "train-02"
        ],
        "sample_groups": {},
        "folds": [
          {
            "fold_id": "native.fold.0",
            "train_sample_ids": [
              "train-01",
              "train-03",
              "train-02"
            ],
            "validation_sample_ids": [
              "train-04",
              "train-06",
              "train-05"
            ],
            "metadata": {}
          },
          {
            "fold_id": "native.fold.1",
            "train_sample_ids": [
              "train-04",
              "train-06",
              "train-05"
            ],
            "validation_sample_ids": [
              "train-01",
              "train-03",
              "train-02"
            ],
            "metadata": {}
          }
        ]
      }
    },
    "generation": {
      "strategy": "cartesian",
      "max_variants": 2,
      "dimensions": [
        {
          "name": "native_model_params",
          "choices": [
            {
              "label": "candidate_0",
              "value": 0,
              "param_overrides": [
                {
                  "node_id": "model:pls",
                  "params": {
                    "alpha": 0.1
                  }
                }
              ],
              "active_subsequence": null
            },
            {
              "label": "candidate_1",
              "value": 1,
              "param_overrides": [
                {
                  "node_id": "model:pls",
                  "params": {
                    "alpha": 1.0
                  }
                }
              ],
              "active_subsequence": null
            }
          ]
        }
      ]
    },
    "shape_plans": {
      "transform:0": {
        "node_id": "transform:0",
        "input_granularity": "sample",
        "target_granularity": "sample",
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "feature_namespace": "spectra",
        "feature_schema_fingerprint": null,
        "target_space": "raw",
        "aggregation_policy": {
          "aggregation_level": "sample",
          "method": "mean",
          "weights": "none",
          "emit_parallel_metrics": true,
          "selection_metric_level": "sample",
          "store_raw_predictions": true,
          "store_aggregated_predictions": true
        },
        "augmentation_policy": {
          "sample_scope": "train_only",
          "feature_scope": "train_only",
          "require_origin_id": true,
          "inherit_group": true,
          "inherit_target": true
        },
        "selection_policy": {
          "scope": "none",
          "store_masks": true,
          "allow_schema_mismatch_on_join": false
        }
      },
      "model:pls": {
        "node_id": "model:pls",
        "input_granularity": "sample",
        "target_granularity": "sample",
        "fit_rows": "fold_train",
        "predict_rows": "fold_validation",
        "feature_namespace": "spectra",
        "feature_schema_fingerprint": null,
        "target_space": "raw",
        "aggregation_policy": {
          "aggregation_level": "sample",
          "method": "mean",
          "weights": "none",
          "emit_parallel_metrics": true,
          "selection_metric_level": "sample",
          "store_raw_predictions": true,
          "store_aggregated_predictions": true
        },
        "augmentation_policy": {
          "sample_scope": "train_only",
          "feature_scope": "train_only",
          "require_origin_id": true,
          "inherit_group": true,
          "inherit_target": true
        },
        "selection_policy": {
          "scope": "none",
          "store_masks": true,
          "allow_schema_mismatch_on_join": false
        }
      }
    },
    "data_bindings": {
      "transform:0": [
        {
          "node_id": "transform:0",
          "input_name": "x",
          "request_id": "io:canonical-raw-pls",
          "schema_fingerprint": "1111111111111111111111111111111111111111111111111111111111111111",
          "plan_fingerprint": "2222222222222222222222222222222222222222222222222222222222222222",
          "relation_fingerprint": "$relation",
          "output_representation": "tabular_numeric",
          "feature_set_id": "spectra",
          "source_ids": [
            "spectra"
          ],
          "require_relations": true,
          "view_policy": {
            "fit_partition": "fold_train",
            "predict_partition": "fold_validation",
            "include_augmented_train": false,
            "include_augmented_validation": false,
            "include_excluded": false,
            "require_sample_ids": true
          },
          "metadata": {}
        }
      ]
    },
    "metadata": {
      "native_pipeline_recipe": {
        "steps": [
          {
            "method_id": "preprocessing.scaling.standard_scale",
            "role": "transformer",
            "params": {}
          },
          {
            "method_id": "models.regularized.ridge",
            "role": "regressor",
            "params": {
              "scale_x": false
            }
          }
        ],
        "candidates": [
          {
            "alpha": 0.1
          },
          {
            "alpha": 1.0
          }
        ]
      },
      "native_pipeline_target_names": [
        "protein",
        "moisture"
      ],
      "input_sample_ids": [
        "train-06",
        "train-05",
        "train-04",
        "train-03",
        "train-02",
        "train-01"
      ]
    }
  },
  "controller_manifests": [],
  "data_identities": [],
  "parameter_patches": [],
  "patch_policies": [],
  "influence_requirements": [],
  "training_losses": [],
  "options": {
    "refit": true,
    "refit_strategy": "refit_one",
    "seed": 91,
    "selection": {
      "id": "selection:rmse",
      "metric": {
        "name": "rmse",
        "objective": "minimize"
      },
      "required_metric_level": "sample",
      "require_finite": true,
      "evaluation_scope": "oof"
    },
    "selection_output_id": "output:prediction",
    "outputs": [
      {
        "output_id": "output:prediction",
        "node_id": "model:pls",
        "port_name": "oof",
        "prediction_level": "sample",
        "unit_level": "physical_sample",
        "prediction_kind": "regression_point",
        "target_names": [
          "protein",
          "moisture"
        ],
        "target_units": [
          null,
          null
        ],
        "class_labels": [
          [],
          []
        ],
        "output_order": "target_order",
        "target_space": "raw"
      }
    ],
    "scheduler": {
      "kind": "sequential",
      "backend": null,
      "workers": 1
    },
    "resources": {
      "cpu_threads": 1,
      "memory_bytes": null,
      "gpu_devices": [],
      "wall_time_ms": null
    },
    "artifacts": {
      "cv_artifacts": "discard",
      "prediction_caches": "retain",
      "fitted_artifacts": "portable_required"
    }
  },
  "request_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000"
}
"#;

fn legacy_request_fixture(facts: &NativeLinearPipelineFacts) -> TrainingRequest {
    let mut value: Value = serde_json::from_str(LEGACY_REQUEST).unwrap();
    value["campaign"]["data_bindings"]["transform:0"][0]["relation_fingerprint"] =
        json!(facts.envelope.relation_fingerprint);
    let binding: DataBinding =
        serde_json::from_value(value["campaign"]["data_bindings"]["transform:0"][0].clone())
            .unwrap();
    value["data_identities"] =
        json!([TrainingDataIdentity::from_binding_envelope(&binding, &facts.envelope).unwrap()]);
    let mut manifests = manifests();
    manifests.sort_by(|a, b| a.controller_id.cmp(&b.controller_id));
    value["controller_manifests"] = json!(manifests);
    let mut request: TrainingRequest = serde_json::from_value(value).unwrap();
    request.request_fingerprint = request.compute_fingerprint().unwrap();
    request
}

#[test]
fn request_fingerprint_matches_frozen_core_request_fields() {
    let facts = facts();
    let actual = native_pipeline_training_request(&facts, &recipe(), &manifests()).unwrap();
    let expected = legacy_request_fixture(&facts);
    expected.validate().unwrap();
    assert_eq!(actual.request_fingerprint, expected.request_fingerprint);
    assert_eq!(json!(actual), json!(expected));
    let mut reversed = manifests();
    reversed.reverse();
    assert_eq!(
        actual.request_fingerprint,
        native_pipeline_training_request(&facts, &recipe(), &reversed)
            .unwrap()
            .request_fingerprint
    );
    let mut changed = facts.clone();
    changed.envelope.data_content_fingerprint = Some("5".repeat(64));
    assert_ne!(
        actual.request_fingerprint,
        native_pipeline_training_request(&changed, &recipe(), &manifests())
            .unwrap()
            .request_fingerprint
    );
}

#[test]
fn default_folds_use_provider_order_and_audit_order_is_only_for_declared_folds() {
    let mut facts = facts();
    facts.declared_folds.clear();
    let actual = native_pipeline_training_request(&facts, &recipe(), &manifests()).unwrap();
    let value = json!(actual);
    assert_eq!(
        value["campaign"]["split_invocation"]["fold_set"]["folds"][0],
        json!({
            "fold_id":"core.fold.0", "train_sample_ids":["train-05","train-03","train-01"],
            "validation_sample_ids":["train-06","train-04","train-02"], "metadata":{}
        })
    );
    facts.input_sample_ids = None;
    assert_eq!(
        actual.request_fingerprint,
        native_pipeline_training_request(&facts, &recipe(), &manifests())
            .unwrap()
            .request_fingerprint
    );
}

#[test]
fn group_origin_and_declared_fold_leakage_policy_stays_with_dag() {
    let mut facts = facts();
    for (i, relation) in facts.relations.records.iter_mut().enumerate() {
        relation.group_id = Some(GroupId::new(format!("g{}", i / 2)).unwrap());
    }
    refresh_relations(&mut facts);
    assert!(native_pipeline_training_request(&facts, &recipe(), &manifests()).is_err());
    // Audit indices place whole groups together: [06,05], [04,03], [02,01].
    facts.declared_folds = vec![
        (vec![1, 2, 4, 5], vec![0, 3]),
        (vec![0, 3], vec![1, 2, 4, 5]),
    ];
    let actual = native_pipeline_training_request(&facts, &recipe(), &manifests()).unwrap();
    assert_eq!(
        json!(actual)["campaign"]["leakage_policy"]["split_unit"],
        "group"
    );
    assert_eq!(
        json!(actual)["campaign"]["leakage_policy"]["require_group_ids"],
        true
    );
    facts.declared_folds.clear();
    assert!(
        native_pipeline_training_request(&facts, &recipe(), &manifests())
            .unwrap_err()
            .contains("explicit leakage-safe fold_ids")
    );

    let mut facts = self::facts();
    facts.relations.records[0].origin_sample_id = Some(SampleId::new("train-01").unwrap());
    refresh_relations(&mut facts);
    // As in Core, request construction carries the policy; execution validates
    // origin relations against folds using the real provider envelope.
    let request = native_pipeline_training_request(&facts, &recipe(), &manifests()).unwrap();
    assert_eq!(
        json!(request)["campaign"]["leakage_policy"]["forbid_origin_cross_fold"],
        true
    );
    assert_eq!(
        json!(request)["campaign"]["leakage_policy"]["split_unit"],
        "group"
    );
    assert!(request
        .campaign
        .validate_data_envelope_relations(&facts.envelope)
        .is_err());
    facts.declared_folds.clear();
    assert!(
        native_pipeline_training_request(&facts, &recipe(), &manifests())
            .unwrap_err()
            .contains("explicit leakage-safe fold_ids")
    );
}

#[test]
fn invalid_training_cohorts_and_declared_indexes_are_refused() {
    for mutation in 0..5 {
        let mut facts = facts();
        match mutation {
            0 => facts.relations.records[0].tags = vec!["partition:test".into()],
            1 => facts.relations.records[0].is_augmented = true,
            2 => facts.relations.records[0].excluded = true,
            3 => {
                facts.relations.records[0].sample_id = facts.relations.records[1].sample_id.clone()
            }
            _ => facts.relations.records.truncate(3),
        }
        refresh_relations(&mut facts);
        assert!(native_pipeline_training_request(&facts, &recipe(), &manifests()).is_err());
    }
    for index in [-1, 6] {
        let mut facts = facts();
        facts.declared_folds[0].0[0] = index;
        assert!(
            native_pipeline_training_request(&facts, &recipe(), &manifests())
                .unwrap_err()
                .contains("declared fold index outside native training cohort")
        );
    }
    let mut facts = facts();
    facts.declared_folds.truncate(1);
    assert!(
        native_pipeline_training_request(&facts, &recipe(), &manifests())
            .unwrap_err()
            .contains("at least two declared CV folds")
    );
}

#[test]
fn target_selection_classification_and_reserved_parameters_keep_reference_policy() {
    let facts = facts();
    let mut selected = recipe();
    selected.steps[1]
        .params
        .insert("native_target_index".into(), json!(1));
    let request = native_pipeline_training_request(&facts, &selected, &manifests()).unwrap();
    assert_eq!(
        json!(request)["options"]["outputs"][0]["target_names"],
        json!(["moisture"])
    );
    assert_eq!(
        json!(request)["campaign"]["metadata"]["native_pipeline_target_names"],
        json!(["protein", "moisture"])
    );
    let mut classification = selected.clone();
    classification.steps[1].role = "classifier".into();
    classification.steps[1].method_id = "fixture.classifier".into();
    let request = native_pipeline_training_request(&facts, &classification, &manifests()).unwrap();
    assert_eq!(
        json!(request)["options"]["outputs"][0]["prediction_kind"],
        "class_label"
    );
    assert_eq!(
        json!(request)["options"]["selection"]["metric"],
        json!({"name":"balanced_accuracy","objective":"maximize"})
    );
    classification.steps[1].params.remove("native_target_index");
    assert!(
        native_pipeline_training_request(&facts, &classification, &manifests())
            .unwrap_err()
            .contains("exactly one target")
    );
    for key in ["method_id", "unsafe_flags", "native_target_index"] {
        let mut recipe = recipe();
        recipe.candidates[0].insert(key.into(), json!(1));
        assert!(
            native_pipeline_training_request(&facts, &recipe, &manifests())
                .unwrap_err()
                .contains("candidate parameters cannot override")
        );
    }
    let mut wrong_role = recipe();
    wrong_role.steps[0].role = "selector".into();
    assert!(
        native_pipeline_training_request(&facts, &wrong_role, &manifests())
            .unwrap_err()
            .contains("native catalog role differs")
    );
    assert!(native_pipeline_training_request(&facts, &recipe(), &[]).is_err());
    let mut no_refit = manifests();
    for manifest in &mut no_refit {
        manifest.supported_phases.remove(&dag_ml_core::Phase::Refit);
    }
    assert!(native_pipeline_training_request(&facts, &recipe(), &no_refit).is_err());
}

#[test]
fn executable_roles_and_native_controller_ids_are_available_without_methods_runtime() {
    assert_eq!(
        METHODS_ESTIMATOR_EXECUTABLE_ROLES,
        [
            N4mRole::Transformer,
            N4mRole::Selector,
            N4mRole::Regressor,
            N4mRole::Classifier,
            N4mRole::SampleFilter
        ]
    );
    assert_eq!(
        methods_native_controller_ids()
            .iter()
            .map(|id| id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "controller:methods.pls",
            "controller:methods.ridge",
            "controller:methods.native.regression",
            "controller:n4m.transformer",
            "controller:n4m.selector",
            "controller:n4m.regressor",
            "controller:n4m.classifier",
            "controller:n4m.sample_filter"
        ]
    );
}
