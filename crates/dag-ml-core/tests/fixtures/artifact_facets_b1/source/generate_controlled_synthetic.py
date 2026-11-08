#!/usr/bin/env python3
"""Controlled SYNTHETIC calibration-fixture dataset generator (not real measurements).

Data-generating rule (documented, deterministic, stdlib only):
  z1, z2 ~ N(0,1) latent factors; band_j = 5 + a_j*z1 + b_j*z2 + 0.15*e_j, e_j ~ N(0,1)
  protein = 3 + 2.0*z1 - 0.4*z2 + 0.3*e_y
Two physically and origin-wise disjoint cohorts are drawn from the SAME rule with
different seeds: 40 'dev.NNN' (source/training) and 48 'cal.NNN' (calibration, held out).
Values are rounded to 4 decimals before being written; the written bytes are the truth.
"""
import json, math, random, sys
from pathlib import Path

A = [0.9, 0.7, 0.5, 0.3, 0.1, -0.2, -0.5, -0.8]
B = [0.1, -0.2, 0.3, -0.1, 0.4, -0.3, 0.2, -0.4]

def gauss(r):
    u1 = 1.0 - r.random(); u2 = r.random()
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

def cohort(prefix, n, seed, partition):
    r = random.Random(seed)
    ids, rows, ys = [], [], []
    for i in range(n):
        z1, z2 = gauss(r), gauss(r)
        rows.append([round(5 + A[j] * z1 + B[j] * z2 + 0.15 * gauss(r), 4) for j in range(len(A))])
        ys.append(round(3 + 2.0 * z1 - 0.4 * z2 + 0.3 * gauss(r), 4))
        ids.append(f"{prefix}.{i:03d}")
    n_f = len(A)
    return {
        "schema": "nirs4all.dataset.v1", "schema_version": 1,
        "dataset": {
            "schema": "nirs4all.multimodal-dataset", "schema_version": 1,
            "name": f"B1 calibration fixture {prefix} (controlled synthetic)",
            "sample_ids": ids, "source_alignment": "strict",
            "sources": [{
                "name": "spectra", "sample_ids": ids, "representation_id": "tabular_numeric",
                "axes": ["sample", "feature"], "feature_names": [f"b{j}" for j in range(n_f)],
                "axis_units": {}, "axis_coordinates": {},
                "array": {"dtype": "float32", "shape": [n, n_f], "values": rows},
                "presence_mask": {"dtype": "bool", "shape": [n], "values": [True] * n},
            }],
            "y": {"dtype": "float32", "shape": [n], "values": ys},
            "groups": None,
            "partitions": {"dtype": "<U5", "shape": [n], "values": [partition] * n},
            "target_names": ["protein"], "task_type": "regression",
            "target_mask": {"dtype": "bool", "shape": [n], "values": [True] * n},
        },
        "origin_ids": ids, "fold_ids": [None] * n,
    }

def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, rec in (("source_dataset.json", cohort("dev", 40, 20261103, "train")),
                      ("calibration_dataset.json", cohort("cal", 48, 20261104, "test"))):
        (out / name).write_text(json.dumps(rec, separators=(",", ":")) + "\n")

if __name__ == "__main__":
    main(sys.argv[1])
