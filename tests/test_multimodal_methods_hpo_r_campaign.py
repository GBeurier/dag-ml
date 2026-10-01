"""Opt-in real public R/CLI/SDK qualification, with a mandatory release mode."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WorkerFailureLifecycleTests(unittest.TestCase):
    """Real child processes must be reaped even when close cannot cooperate."""

    def test_invalid_ack_and_timeout_reap_sigterm_resistant_workers(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "r_role_qualification", ROOT / "scripts/qualify_multimodal_methods_hpo_r.py"
        )
        assert spec is not None and spec.loader is not None
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        for mode in ("invalid_ack", "timeout", "primary_error"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                work = Path(directory)
                adapter = work / "uncooperative-worker"
                adapter.write_text(
                    f"#!{sys.executable}\n"
                    "import json, signal, sys, time\n"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                    "for line in sys.stdin:\n"
                    " frame = json.loads(line)\n"
                    " if frame['type'] == 'init':\n"
                    "  print(json.dumps({'type':'ack','schema_version':1,'status':'initialized'}), flush=True)\n"
                    " elif frame['type'] == 'close':\n"
                    + ("  print(json.dumps({'type':'ack','schema_version':1,'status':'wrong'}), flush=True)\n"
                       if mode != "timeout" else "  pass\n")
                    + "  while True: time.sleep(1)\n"
                    "while True: time.sleep(1)\n"
                )
                adapter.chmod(0o755)
                worker = campaign.RWorker({"adapter": str(adapter), "workdir": str(work)}, timeout=5)
                worker.timeout = 0.15
                if mode == "primary_error":
                    with self.assertRaisesRegex(RuntimeError, "primary controller failure"), worker:
                        raise RuntimeError("primary controller failure")
                else:
                    expected = TimeoutError if mode == "timeout" else AssertionError
                    with self.assertRaises(expected):
                        worker.close()
                self.assertIsNotNone(worker.process.poll(), "Uncooperative child still running")
                self.assertEqual(worker.process.returncode, -9, "SIGTERM-resistant child was not killed/reaped")
                self.assertTrue(worker.process.stdin.closed)
                self.assertTrue(worker.process.stdout.closed)
                self.assertTrue(worker._stderr_stream.closed)
                self.assertFalse(worker._reader.is_alive())


class MultimodalMethodsRQualificationTests(unittest.TestCase):
    def test_actual_r_cli_hpo_and_five_model_archive_replay(self) -> None:
        capture = os.environ.get("NIRS4ALL_R_ROLE_NODE_CAPTURE")
        cli = os.environ.get("DAG_ML_CLI")
        if not capture or not cli:
            if os.environ.get("NIRS4ALL_REQUIRE_R_ROLE_CAMPAIGN") == "1":
                self.fail("Mandatory R qualification needs NIRS4ALL_R_ROLE_NODE_CAPTURE and DAG_ML_CLI")
            self.skipTest("Set fresh Node capture and native CLI paths for real R qualification")
        with tempfile.TemporaryDirectory(prefix="dagml-r-role-qualification-") as directory:
            work = Path(directory) / "campaign"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/qualify_multimodal_methods_hpo_r.py"),
                 "--node-capture", str(capture), "--cli", str(cli), "--workdir", str(work)],
                cwd=ROOT, capture_output=True, text=True, timeout=1800, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)
            proof = json.loads((work / "qualification-receipt.json").read_text())
            self.assertEqual(proof["status"], "passed")
            self.assertEqual(proof["cli_hpo_trials"], 3)
            self.assertEqual(proof["actual_raw_models"], 5)
            self.assertEqual(proof["fresh_r_replay"]["r_lifecycle"]["hydrate"], 5)
            self.assertEqual(proof["fresh_r_replay"]["r_lifecycle"].get("fit", 0), 0)
            self.assertTrue(proof["independent_source_and_target_row_permutations"])


if __name__ == "__main__":
    unittest.main()
