#!/usr/bin/env python3
"""Generate a leakage-safe sklearn OOF campaign fixture.

This demonstrator intentionally stays outside nirs4all. It exercises the
coordinator contracts with a complex sklearn workflow:

- repeated observations per sample;
- group-aware sample splits;
- train-only sample augmentation;
- multiple branch-specific model variants;
- sample-level aggregation of repeated observation predictions;
- heterogeneous merge variants using OOF predictions plus original data;
- OOF-based model/merge selection followed by final refit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "examples" / "generated"


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    producer_node: str
    estimator: Pipeline


@dataclass(frozen=True)
class MergeSpec:
    producer_node: str
    original_feature_mode: str
    estimator: Pipeline


def sample_id(idx: int) -> str:
    return f"sample:{idx:03d}"


def group_id(idx: int) -> str:
    return f"group:{idx:02d}"


def observation_id(sample_idx: int, rep_idx: int) -> str:
    return f"obs:{sample_idx:03d}:rep:{rep_idx}"


def build_dataset(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n_samples = 60
    repetitions = 3
    n_spectral = 64
    n_meta = 4

    grid = np.linspace(0.0, 1.0, n_spectral)
    latent = rng.normal(size=(n_samples, 5))
    groups = np.array([idx // 5 for idx in range(n_samples)])
    group_effect = np.sin(groups * 0.7) * 0.8
    y = (
        1.7 * latent[:, 0]
        - 1.1 * latent[:, 1]
        + 0.8 * latent[:, 2] ** 2
        + group_effect
        + rng.normal(0.0, 0.08, n_samples)
    )

    sample_features = []
    for idx in range(n_samples):
        center = 0.18 + 0.025 * groups[idx] + 0.03 * latent[idx, 0]
        width = 0.04 + 0.008 * abs(latent[idx, 1])
        peak = np.exp(-((grid - center) ** 2) / (2.0 * width**2))
        harmonic = np.sin(grid * (4.0 + latent[idx, 2]) * math.pi)
        shoulder = np.cos(grid * (2.0 + groups[idx] * 0.2) * math.pi)
        spectral = (
            0.9 * peak
            + 0.25 * harmonic
            + 0.18 * shoulder
            + 0.08 * latent[idx, 3] * grid
        )
        meta = np.array(
            [
                groups[idx] / 11.0,
                latent[idx, 0],
                latent[idx, 1] * latent[idx, 2],
                np.linalg.norm(latent[idx, :3]),
            ]
        )
        sample_features.append((spectral, meta))

    rows = []
    row_sample_idx = []
    row_rep_idx = []
    relations = []
    for sample_idx, (spectral, meta) in enumerate(sample_features):
        for rep_idx in range(repetitions):
            rep_noise = rng.normal(0.0, 0.025, n_spectral)
            rep_shift = rng.normal(0.0, 0.01)
            rows.append(np.concatenate([spectral + rep_noise + rep_shift, meta]))
            row_sample_idx.append(sample_idx)
            row_rep_idx.append(rep_idx)
            relations.append(
                {
                    "observation_id": observation_id(sample_idx, rep_idx),
                    "sample_id": sample_id(sample_idx),
                    "target_id": "target:y",
                    "group_id": group_id(groups[sample_idx]),
                    "origin_sample_id": None,
                    "source_id": "synthetic:sklearn",
                    "is_augmented": False,
                }
            )

    return {
        "x_obs": np.asarray(rows, dtype=np.float64),
        "y_sample": y.astype(np.float64),
        "row_sample_idx": np.asarray(row_sample_idx, dtype=np.int64),
        "row_rep_idx": np.asarray(row_rep_idx, dtype=np.int64),
        "sample_groups": groups.astype(np.int64),
        "relations": relations,
        "n_spectral": n_spectral,
        "n_meta": n_meta,
    }


def branch_specs(seed: int) -> list[BranchSpec]:
    return [
        BranchSpec(
            "branch:b0",
            "branch:b0.variant:pca10_ridge_a03",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=10, random_state=seed)),
                    ("ridge", Ridge(alpha=0.3)),
                ]
            ),
        ),
        BranchSpec(
            "branch:b0",
            "branch:b0.variant:pca16_ridge_a12",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=16, random_state=seed + 1)),
                    ("ridge", Ridge(alpha=1.2)),
                ]
            ),
        ),
        BranchSpec(
            "branch:b1",
            "branch:b1.variant:rf_select_k28",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("select", SelectKBest(f_regression, k=28)),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=80,
                            min_samples_leaf=2,
                            random_state=seed + 11,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        BranchSpec(
            "branch:b1",
            "branch:b1.variant:rf_select_k40",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("select", SelectKBest(f_regression, k=40)),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=120,
                            min_samples_leaf=1,
                            max_features="sqrt",
                            random_state=seed + 12,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        BranchSpec(
            "branch:b2",
            "branch:b2.variant:poly_extra_k45",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("select", SelectKBest(f_regression, k=45)),
                    (
                        "extra",
                        ExtraTreesRegressor(
                            n_estimators=96,
                            min_samples_leaf=2,
                            random_state=seed + 23,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        BranchSpec(
            "branch:b2",
            "branch:b2.variant:poly_extra_k80",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("select", SelectKBest(f_regression, k=80)),
                    (
                        "extra",
                        ExtraTreesRegressor(
                            n_estimators=128,
                            min_samples_leaf=1,
                            random_state=seed + 24,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def merge_specs(seed: int) -> list[MergeSpec]:
    return [
        MergeSpec(
            "merge:m0.pred_only.meta:ridge",
            "none",
            Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=0.4))]),
        ),
        MergeSpec(
            "merge:m1.pred_meta_original.meta:ridge",
            "metadata",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", Ridge(alpha=0.2)),
                ]
            ),
        ),
        MergeSpec(
            "merge:m2.pred_raw_select.meta:extra",
            "all",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("select", SelectKBest(f_regression, k=24)),
                    (
                        "extra",
                        ExtraTreesRegressor(
                            n_estimators=96,
                            min_samples_leaf=2,
                            random_state=seed + 102,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
    ]


def augment_train(
    x_train: np.ndarray, y_train: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.018, size=x_train.shape)
    # Metadata columns are already low-dimensional descriptors; keep them stable.
    if x_train.shape[1] > 8:
        noise[:, -4:] = 0.0
    x_aug = x_train + noise
    return np.vstack([x_train, x_aug]), np.concatenate([y_train, y_train]), x_aug.shape[0]


def aggregate_by_sample(
    row_sample_idx: np.ndarray,
    prediction_rows: np.ndarray,
    validation_samples: np.ndarray,
) -> np.ndarray:
    values = []
    for sample_idx in validation_samples:
        sample_preds = prediction_rows[row_sample_idx == sample_idx]
        if sample_preds.size == 0:
            raise RuntimeError(f"missing validation predictions for sample {sample_idx}")
        values.append(float(np.mean(sample_preds)))
    return np.asarray(values, dtype=np.float64)


def sample_mean_features(x_obs: np.ndarray, row_sample_idx: np.ndarray, n_samples: int) -> np.ndarray:
    rows = []
    for sample_idx in range(n_samples):
        sample_rows = x_obs[row_sample_idx == sample_idx]
        if sample_rows.size == 0:
            raise RuntimeError(f"missing observations for sample {sample_idx}")
        rows.append(np.mean(sample_rows, axis=0))
    return np.asarray(rows, dtype=np.float64)


def metric_bundle(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def merge_features(
    spec: MergeSpec,
    selected_base_oof: np.ndarray,
    raw_sample_features: np.ndarray,
) -> np.ndarray:
    if spec.original_feature_mode == "all":
        return np.hstack([selected_base_oof, raw_sample_features])
    if spec.original_feature_mode == "metadata":
        return np.hstack([selected_base_oof, raw_sample_features[:, -4:]])
    return selected_base_oof


def select_best_by_rmse(metrics: dict[str, dict[str, float]], candidates: list[str]) -> str:
    return min(candidates, key=lambda node: (metrics[node]["rmse"], node))


def run_demo(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    data = build_dataset(seed)
    x_obs = data["x_obs"]
    y_sample = data["y_sample"]
    row_sample_idx = data["row_sample_idx"]
    sample_groups = data["sample_groups"]
    n_spectral = int(data["n_spectral"])

    samples = np.arange(len(y_sample), dtype=np.int64)
    fold_iter = list(GroupKFold(n_splits=5).split(samples, y_sample, groups=sample_groups))
    branches = branch_specs(seed)
    branch_by_node = {branch.producer_node: branch for branch in branches}
    branch_oof = {
        branch.producer_node: np.zeros(len(samples), dtype=np.float64)
        for branch in branches
    }
    prediction_blocks: list[dict[str, Any]] = []
    fold_specs: list[dict[str, Any]] = []
    augmentation_counts: dict[str, int] = {}

    for fold_idx, (train_samples, val_samples) in enumerate(fold_iter):
        fold_name = f"fold:{fold_idx}"
        train_sample_set = set(int(idx) for idx in train_samples)
        val_sample_set = set(int(idx) for idx in val_samples)
        train_rows = np.array([idx in train_sample_set for idx in row_sample_idx])
        val_rows = np.array([idx in val_sample_set for idx in row_sample_idx])
        x_train = x_obs[train_rows]
        y_train = y_sample[row_sample_idx[train_rows]]
        x_val = x_obs[val_rows]
        val_row_sample_idx = row_sample_idx[val_rows]

        x_fit, y_fit, n_augmented = augment_train(x_train, y_train, seed + 1000 + fold_idx)
        augmentation_counts[fold_name] = n_augmented

        fold_specs.append(
            {
                "fold_id": fold_name,
                "train_sample_ids": [sample_id(int(idx)) for idx in train_samples],
                "validation_sample_ids": [sample_id(int(idx)) for idx in val_samples],
                "metadata": {
                    "splitter": "sklearn.model_selection.GroupKFold",
                    "n_train_observations": int(train_rows.sum()),
                    "n_validation_observations": int(val_rows.sum()),
                    "n_train_augmented_observations": int(n_augmented),
                },
            }
        )

        for branch in branches:
            estimator = clone(branch.estimator)
            estimator.fit(x_fit[:, :n_spectral], y_fit)
            val_obs_pred = estimator.predict(x_val[:, :n_spectral])
            val_sample_pred = aggregate_by_sample(
                val_row_sample_idx, val_obs_pred, val_samples
            )
            branch_oof[branch.producer_node][val_samples] = val_sample_pred
            prediction_blocks.append(
                {
                    "prediction_id": f"{branch.producer_node}:fold:{fold_idx}",
                    "producer_node": branch.producer_node,
                    "partition": "validation",
                    "fold_id": fold_name,
                    "sample_ids": [sample_id(int(idx)) for idx in val_samples],
                    "values": [[float(value)] for value in val_sample_pred],
                    "target_names": ["y"],
                }
            )

    branch_metrics = {
        node: metric_bundle(y_sample, values) for node, values in branch_oof.items()
    }
    branch_ids = sorted({branch.branch_id for branch in branches})
    selected_branch_nodes = {
        branch_id: select_best_by_rmse(
            branch_metrics,
            [
                branch.producer_node
                for branch in branches
                if branch.branch_id == branch_id
            ],
        )
        for branch_id in branch_ids
    }
    selected_base_oof = np.column_stack(
        [branch_oof[selected_branch_nodes[branch_id]] for branch_id in branch_ids]
    )
    raw_sample_features = sample_mean_features(x_obs, row_sample_idx, len(samples))

    merge_oof: dict[str, np.ndarray] = {}
    for merge in merge_specs(seed):
        all_merge_features = merge_features(merge, selected_base_oof, raw_sample_features)
        merge_oof[merge.producer_node] = np.zeros(len(samples), dtype=np.float64)
        for fold_idx, (train_samples, val_samples) in enumerate(fold_iter):
            estimator = clone(merge.estimator)
            estimator.fit(all_merge_features[train_samples], y_sample[train_samples])
            meta_pred = estimator.predict(all_merge_features[val_samples])
            merge_oof[merge.producer_node][val_samples] = meta_pred
            prediction_blocks.append(
                {
                    "prediction_id": f"{merge.producer_node}:fold:{fold_idx}",
                    "producer_node": merge.producer_node,
                    "partition": "validation",
                    "fold_id": f"fold:{fold_idx}",
                    "sample_ids": [sample_id(int(idx)) for idx in val_samples],
                    "values": [[float(value)] for value in meta_pred],
                    "target_names": ["y"],
                }
            )

    merge_metrics = {
        node: metric_bundle(y_sample, values) for node, values in merge_oof.items()
    }
    selected_merge_node = select_best_by_rmse(merge_metrics, list(merge_oof.keys()))
    selected_merge_spec = next(
        spec for spec in merge_specs(seed) if spec.producer_node == selected_merge_node
    )

    # Final refit contract:
    # - selected base models are refit on all observations plus train-only style
    #   augmentation over the full training set;
    # - selected merge model is fitted on training meta-features built from OOF
    #   base predictions plus original data if the selected merge variant uses it.
    full_y_obs = y_sample[row_sample_idx]
    x_full_fit, y_full_fit, n_full_augmented = augment_train(x_obs, full_y_obs, seed + 9000)
    refit_base_nodes = []
    for node in selected_branch_nodes.values():
        estimator = clone(branch_by_node[node].estimator)
        estimator.fit(x_full_fit[:, :n_spectral], y_full_fit)
        refit_base_nodes.append(node)
    selected_merge_features = merge_features(
        selected_merge_spec, selected_base_oof, raw_sample_features
    )
    refit_merge = clone(selected_merge_spec.estimator)
    refit_merge.fit(selected_merge_features, y_sample)

    sample_ids = [sample_id(int(idx)) for idx in samples]
    campaign = {
        "fold_set": {
            "id": "sklearn_complex_group_oof",
            "sample_ids": sample_ids,
            "folds": fold_specs,
            "sample_groups": {
                sample_id(int(idx)): group_id(int(sample_groups[idx])) for idx in samples
            },
        },
        "join_policy": {
            "node_id": "merge:heterogeneous_stack_features",
            "allow_train_predictions_as_features": False,
            "join_on": "sample_id",
            "include_partitions": ["validation"],
        },
        "requested_sample_order": sample_ids,
        "prediction_blocks": prediction_blocks,
    }

    report = {
        "seed": seed,
        "sklearn_workflow": {
            "splitter": "GroupKFold(n_splits=5)",
            "repetitions_per_sample": 3,
            "augmentation": "train_only_gaussian_noise_one_copy_per_train_observation",
            "branch_input": "spectral source only",
            "branch_variants": {
                branch_id: [
                    branch.producer_node
                    for branch in branches
                    if branch.branch_id == branch_id
                ]
                for branch_id in branch_ids
            },
            "heterogeneous_merge_variants": [
                {
                    "producer_node": spec.producer_node,
                    "inputs": "selected_oof_predictions+original_sample_features"
                    if spec.original_feature_mode == "all"
                    else "selected_oof_predictions+original_metadata_features"
                    if spec.original_feature_mode == "metadata"
                    else "selected_oof_predictions",
                }
                for spec in merge_specs(seed)
            ],
        },
        "sample_count": int(len(samples)),
        "observation_count": int(len(row_sample_idx)),
        "original_sample_feature_shape": list(raw_sample_features.shape),
        "augmentation_counts": augmentation_counts,
        "branch_variant_metrics": branch_metrics,
        "selected_branch_variants": {
            branch_id: {
                "producer_node": node,
                "selection_metric": "sample_oof_rmse",
                "score": branch_metrics[node],
            }
            for branch_id, node in selected_branch_nodes.items()
        },
        "merge_variant_metrics": merge_metrics,
        "selected_merge_variant": {
            "producer_node": selected_merge_node,
            "selection_metric": "sample_oof_rmse",
            "score": merge_metrics[selected_merge_node],
            "original_feature_mode": selected_merge_spec.original_feature_mode,
        },
        "final_refit": {
            "selected_base_nodes": refit_base_nodes,
            "selected_merge_node": selected_merge_node,
            "base_refit_training_observations": int(len(x_full_fit)),
            "base_refit_augmented_observations": int(n_full_augmented),
            "merge_refit_samples": int(len(samples)),
            "merge_refit_features": int(selected_merge_features.shape[1]),
            "meta_training_features": "OOF base predictions from selected branches plus original sample features"
            if selected_merge_spec.original_feature_mode == "all"
            else "OOF base predictions from selected branches plus original metadata features"
            if selected_merge_spec.original_feature_mode == "metadata"
            else "OOF base predictions from selected branches",
        },
        "leakage_controls": {
            "split_unit": "sample",
            "group_boundary": "GroupKFold over sample groups",
            "validation_augmentation": "disabled",
            "branch_selection": "OOF sample RMSE only",
            "merge_selection": "OOF sample RMSE only",
            "stacking_features": "base model validation OOF predictions only",
            "heterogeneous_merge": "selected OOF predictions plus original sample features; supervised merge steps fit only on fold train",
            "aggregation": "mean prediction over repeated observations per sample",
            "refit": "base models refit on full train; merge model refit on OOF meta-features and selected score policy",
        },
    }
    return campaign, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20260526)
    args = parser.parse_args()

    campaign, report = run_demo(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    campaign_path = args.out_dir / "sklearn_complex_oof_campaign.json"
    report_path = args.out_dir / "sklearn_complex_report.json"
    campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {campaign_path}")
    print(f"wrote {report_path}")
    print(
        "selected merge={producer_node}, rmse={rmse:.6f}, r2={r2:.6f}".format(
            producer_node=report["selected_merge_variant"]["producer_node"],
            **report["selected_merge_variant"]["score"],
        )
    )


if __name__ == "__main__":
    main()
