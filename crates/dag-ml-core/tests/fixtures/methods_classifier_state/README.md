# Genuine Methods multimodal classifier states

Each JSON file holds the exact bytes of
`n4m.MultimodalClassifierPipeline.export_state()` for the closed four-source
DAG-ML classifier recipe (`standard_scaler`, two `tensor_pca`, one
`column_transformer`, three classes), together with the recipe, source schemas
and class labels that sign it. `export_states.py` produced them unchanged from
the published PyPI wheels on Linux x86-64, with seeded synthetic data:

| File | nirs4all-methods | Writer ABI | Nested `standard_scale` parameters |
| --- | --- | --- | --- |
| `methods_1_3_14.json` | 1.3.14 | 2.20.0 | `with_mean`, `with_std` |
| `methods_1_3_15.json` | 1.3.15 | 2.22.0 | `with_mean`, `with_std`, `ddof=0` |
| `methods_1_3_18.json` | 1.3.18 | 2.23.0 | `with_mean`, `with_std`, `ddof=0` |

They witness the bounded wire validator in
`src/methods_classification_state.rs` across the release that added `ddof` to
the scaler state. They are structural wire evidence, not numerical
qualification: Methods owns hydration and prediction.
