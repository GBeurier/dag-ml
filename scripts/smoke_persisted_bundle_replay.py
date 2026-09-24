#!/usr/bin/env python3
"""Prove CV bundle replay loads signed host sidecars after a process restart."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples" / "adapters"))
from python_process_controller import stable_handle  # noqa: E402


def run(cli: Path, arguments: list[str], sidecars: Path, *, succeeds: bool = True) -> None:
    env = os.environ.copy()
    env["DAG_ML_PROCESS_ARTIFACT_DIR"] = str(sidecars)
    completed = subprocess.run(
        [str(cli), *arguments], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if succeeds and completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    if not succeeds and completed.returncode == 0:
        raise AssertionError("tampered replay unexpectedly succeeded")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: smoke_persisted_bundle_replay.py PATH_TO_DAG_ML_CLI")
    cli = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="dagml-persisted-replay-") as temp:
        work = Path(temp)
        sidecars = work / "sidecars"
        bundle_path = work / "bundle.json"
        cache_path = work / "cache.json"
        request_path = work / "request.json"
        handles_path = work / "handles.json"
        output_path = work / "replay.json"
        run(cli, [
            "run-process-cv-refit-bundle",
            "--graph", "examples/branch_merge_oof_graph.json",
            "--campaign", "examples/campaign_branch_merge_oof.json",
            "--controllers", "examples/controller_manifests.json",
            "--envelope", "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter", "examples/adapters/persisted_sidecar_process_controller.py",
            "--selections", "examples/fixtures/bundle/selection_decisions_branch_merge.json",
            "--bundle-id", "bundle:persisted.replay.smoke",
            "--plan-id", "plan:persisted.replay.smoke",
            "--output", str(bundle_path),
            "--prediction-cache-output", str(cache_path),
        ], sidecars)
        bundle = json.loads(bundle_path.read_text())
        records = bundle["refit_artifacts"]
        assert len(records) == 3
        request = json.loads(
            (ROOT / "examples/fixtures/bundle/replay_request_branch_merge_predict.json").read_text()
        )
        request["bundle_id"] = bundle["bundle_id"]
        request_path.write_text(json.dumps(request))
        handles = {
            record["artifact"]["id"]: {
                "handle": stable_handle(record["artifact"]["id"]),
                "kind": "model",
                "owner_controller": record["controller_id"],
            }
            for record in records
        }
        handles_path.write_text(json.dumps(handles))
        replay = [
            "run-process-replay",
            "--graph", "examples/branch_merge_oof_graph.json",
            "--campaign", "examples/campaign_branch_merge_oof.json",
            "--controllers", "examples/controller_manifests.json",
            "--bundle", str(bundle_path),
            "--artifact-handles", str(handles_path),
            "--replay-request", str(request_path),
            "--adapter", "examples/adapters/persisted_sidecar_process_controller.py",
            "--plan-id", "plan:persisted.replay.smoke",
            "--output", str(output_path),
        ]
        for key in request["data_envelope_keys"]:
            replay.extend([
                "--envelope",
                f"{key}=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            ])
        # This starts a second CLI and fresh one-shot operator processes.
        run(cli, replay, sidecars)
        outcome = json.loads(output_path.read_text())
        assert outcome["bundle_id"] == bundle["bundle_id"]
        assert len(outcome["node_results"]) == len(records)
        assert len(outcome["prediction_blocks"]) == len(records)
        by_node = {block["producer_node"]: block for block in outcome["prediction_blocks"]}
        for record in records:
            block = by_node[record["node_id"]]
            sidecar = sidecars / Path(record["artifact"]["uri"]).name
            expected = json.loads(sidecar.read_text())["prediction_value"]
            assert all(row == [expected] for row in block["values"])

        missing = handles.copy()
        missing.pop(next(iter(missing)))
        handles_path.write_text(json.dumps(missing))
        run(cli, replay, sidecars, succeeds=False)
        handles_path.write_text(json.dumps(handles))

        sidecar = sidecars / Path(records[0]["artifact"]["uri"]).name
        sidecar.write_bytes(sidecar.read_bytes() + b" ")
        run(cli, replay, sidecars, succeeds=False)
    print("persistent CV bundle replay and tamper refusals passed")


if __name__ == "__main__":
    main()
