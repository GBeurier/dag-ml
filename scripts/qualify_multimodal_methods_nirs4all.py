"""Qualify the public Python SDK on the four-source numeric Methods fixture.

This checks SDK availability, native scheduling and fit-free archive replay.
Exact HPO trial/score parity is checked separately by the callback oracle.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("N4A_DAGML_INPROCESS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import nirs4all
import numpy as np
from n4m.roles import RolePipeline
from nirs4all.pipeline.runner import PipelineRunner
from nirs4all_io import MultimodalDataset, TensorSource
from sklearn.model_selection import GroupKFold


def main() -> None:
    fixture = json.loads(Path(sys.argv[1]).read_text())
    ids = fixture["sampleIds"]
    sources = {
        name: TensorSource(np.asarray(rows), ids, representation_id="tabular_numeric")
        for name, rows in fixture["sourceRows"].items()
    }
    cohort = MultimodalDataset(
        sources,
        sample_ids=ids,
        y=fixture["target"],
        groups=["plant-" + str(i // 2) for i in range(len(ids))],
        partitions=["train"] * len(ids),
        name="four-source-methods-qualification",
    )

    def model():
        return RolePipeline.from_steps(
            [
                {"class": "n4m:models.regularized.ridge", "params": {"alpha": 0.2}},
            ]
        )

    pipeline = [
        GroupKFold(3),
        {"branch": {"by_source": True, "steps": [{"model": model()}]}},
        {"merge": "predictions"},
        {"model": model()},
    ]
    with tempfile.TemporaryDirectory(
        prefix="nirs4all-methods-four-source-"
    ) as workspace:
        with patch.object(
            PipelineRunner,
            "run",
            side_effect=AssertionError("Legacy execution forbidden"),
        ):
            result = nirs4all.run(
                pipeline,
                cohort,
                engine="dag-ml",
                refit=True,
                save_artifacts=False,
                save_charts=False,
                verbose=0,
                random_state=19,
                workspace_path=workspace,
            )
        try:
            assert len(result.runs) == 5
            archive = result.runs[-1].export(Path(workspace) / "selected.n4a")
            with patch.object(
                RolePipeline, "fit", side_effect=AssertionError("Replay must not fit")
            ):
                replay = nirs4all.predict(archive, cohort)
                again = nirs4all.predict(archive, cohort)
            prediction = replay.to_numpy()
            assert prediction.shape[0] == len(ids)
            assert np.isfinite(prediction).all()
            np.testing.assert_array_equal(prediction, again.to_numpy())
            receipt = {
                "nirs4all_version": nirs4all.__version__,
                "sources": list(sources),
                "runs": len(result.runs),
                "samples": len(ids),
                "predictions": prediction.tolist(),
                "replay_fit_forbidden": True,
                "legacy_training_forbidden": True,
            }
        finally:
            result.close()
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(receipt, indent=2) + "\n")
    print("NIRS4ALL_MULTIMODAL_METHODS_REPLAY_OK", receipt["runs"], receipt["samples"])


if __name__ == "__main__":
    main()
