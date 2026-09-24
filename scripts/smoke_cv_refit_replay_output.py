#!/usr/bin/env python3
"""Exercise the native one-session CV, REFIT and PREDICT JSON contract."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_cv_refit_replay_output.py PATH_TO_DAG_ML_CLI")
    cli = Path(sys.argv[1]).resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="dagml-cv-refit-replay-") as temp:
        outcome_path = Path(temp) / "outcome.json"
        subprocess.run(
            [
                str(cli), "run-process-dsl-cv-refit-replay",
                "--dsl", "examples/pipeline_dsl_branch_merge_executable.json",
                "--controllers", "examples/controller_manifests.json",
                "--envelope", "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
                "--adapter", "examples/adapters/python_process_controller.py",
                "--selections", "examples/fixtures/bundle/selection_decisions_branch_merge.json",
                "--output", str(outcome_path),
            ],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,
        )
        outcome = json.loads(outcome_path.read_text())
    assert outcome["bundle"]["bundle_id"]
    assert outcome["fit_cv_result_count"] > 0
    assert outcome["refit_result_count"] > 0
    assert outcome["replay_node_results"]
    blocks = outcome["replay_prediction_blocks"]
    assert blocks and any(block["sample_ids"] and block["values"] for block in blocks)
    assert "replay_scores" in outcome
    print("CV -> REFIT -> PREDICT native JSON outcome passed")


if __name__ == "__main__":
    main()
