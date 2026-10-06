"""Storage bridge from validated DAG experiment views to SDK SQLite/Parquet.

No prediction, metric, model selection or fit is performed by this adapter.
The SDK store owns its schema and array writer; DAG retains native identities.
"""
from __future__ import annotations

from typing import Any


def publish_experiment_to_sdk(store: Any, view: dict[str, Any], *, native_run_id: str,
                              experiment_path: str, winner_variant_id: str) -> str:
    """Persist exact native prediction arrays/provenance through WorkspaceStore."""
    rows, scores = view["predictions"], view["score_set"]
    run = store.begin_run(native_run_id, {"engine": "dag-ml", "native_run_id": native_run_id,
                                        "native_experiment": experiment_path}, [])
    pipelines: dict[str, str] = {}
    chains: dict[tuple[str, str], str] = {}
    arrays = []
    try:
        with store.transaction():
            for index, row in enumerate(rows):
                variant = row["variant_id"]
                label = row["dataset"]
                if variant not in pipelines:
                    pipelines[variant] = store.begin_pipeline(run, variant, {"native_variant_id": variant}, [], label, "")
                chain_key = (variant, row["model_name"])
                if chain_key not in chains:
                    chains[chain_key] = store.save_chain(pipelines[variant], [], -1, row["model_name"], "",
                                                        "native_results", {}, {}, dataset_name=label)
                prediction = store.save_prediction(
                    pipelines[variant], chains[chain_key], label, row["model_name"], row["model_name"],
                    row["fold_id"], row["partition"], row.get("val_score"), row.get("test_score"),
                    row.get("train_score"), row["metric"], row["task_type"], len(row["sample_indices"]), 0,
                    row.get("scores", {}), {}, None, None, 0, 0.0, refit_context=row["refit_context"])
                metadata = {"schema": "dagml.sdk-workspace-row.v1", "native_run_id": native_run_id,
                            "native_row_index": index, "variant_id": variant, "sample_ids": row["sample_ids"],
                            "target_names": row.get("target_names", []), "native_score_set": scores,
                            "shapes": {key: row[key + "_shape"] for key in ("y_true", "y_pred", "y_proba")}}
                import numpy as np
                arrays.append({"prediction_id": prediction, "dataset_name": label, "model_name": row["model_name"],
                               "fold_id": row["fold_id"], "partition": row["partition"], "metric": row["metric"],
                               "val_score": row.get("val_score"), "task_type": row["task_type"],
                               "y_true": np.asarray(row["y_true"]).reshape(row["y_true_shape"]) if row["y_true_shape"] else None,
                               "y_pred": np.asarray(row["y_pred"]).reshape(row["y_pred_shape"]),
                               "y_proba": np.asarray(row["y_proba"]).reshape(row["y_proba_shape"]) if row["y_proba_shape"] else None,
                               "sample_indices": np.asarray(row["sample_indices"]), "weights": np.asarray(row.get("weights", [])),
                               "sample_metadata": None, "result_metadata": metadata})
            store.array_store.save_batch(arrays)
            for pipeline in pipelines.values():
                store.complete_pipeline(pipeline, None, None, scores.get("selection_metric", ""), 0)
            store.complete_run(run, {"native_run_id": native_run_id, "native_experiment": experiment_path,
                                     "winner_variant_id": winner_variant_id, "native_score_set": scores})
    except BaseException as error:
        store.fail_run(run, str(error))
        raise
    return run
