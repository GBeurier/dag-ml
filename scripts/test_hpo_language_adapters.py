#!/usr/bin/env python3
"""Check the JSONL HPO host contract across installed language runtimes."""

import json
import os
import select
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "examples" / "adapters"
EVENTS = [
    ({"operation": "init"}, {"prepared_checkpoint": None, "interrupted": []}),
    ({"operation": "ask", "trial_index": 0}, {"params": {"n_components": 1}}),
    ({"operation": "ask", "trial_index": 3}, {"params": {"n_components": 4}}),
    ({"operation": "report_intermediate", "trial_index": 0, "fold_index": 0, "score": 1.2}, {"prune": False}),
    ({"operation": "tell", "trial_index": 0, "score": 1.2}, {"ok": True}),
    ({"operation": "pruned", "trial_index": 1}, {"ok": True}),
    ({"operation": "fail", "trial_index": 2}, {"ok": True}),
    ({"operation": "prepare_terminal", "checkpoint": {}}, {"ok": True}),
    ({"operation": "checkpoint", "checkpoint": {}}, {"ok": True}),
    ({"operation": "unsupported"}, {"error": "unsupported HPO operation unsupported"}),
]


class HostHpoLanguageAdapterTests(unittest.TestCase):
    def check_adapter(self, command):
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        try:
            for event, expected in EVENTS:
                process.stdin.write(json.dumps(event) + "\n")
                process.stdin.flush()
                ready, _, _ = select.select([process.stdout], [], [], 30)
                self.assertTrue(ready, f"{command[0]} did not reply to {event['operation']}")
                reply = process.stdout.readline()
                self.assertTrue(reply, f"{command[0]} exited during {event['operation']}")
                self.assertEqual(json.loads(reply), expected)
            process.stdin.close()
            return_code = process.wait(timeout=30)
            self.assertEqual(return_code, 0, process.stderr.read())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=30)
            process.stdout.close()
            process.stderr.close()
            if not process.stdin.closed:
                process.stdin.close()

    def test_python_reference(self):
        self.check_adapter([sys.executable, str(ADAPTERS / "hpo_optimizer_jsonl.py")])

    def test_r_jsonl(self):
        if not shutil.which("Rscript"):
            if os.environ.get("DAGML_REQUIRE_HPO_R") == "1":
                self.fail("Rscript is required for R HPO adapter parity")
            self.skipTest("Rscript is not installed")
        self.check_adapter(["Rscript", str(ADAPTERS / "hpo_optimizer_jsonl.R")])

    def test_matlab_or_octave_jsonl(self):
        if not (shutil.which("octave") or shutil.which("matlab")):
            if os.environ.get("DAGML_REQUIRE_HPO_MATLAB") == "1":
                self.fail("Octave or MATLAB is required for HPO adapter parity")
            self.skipTest("Octave/MATLAB is not installed")
        self.check_adapter([str(ADAPTERS / "hpo_optimizer_matlab.sh")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
