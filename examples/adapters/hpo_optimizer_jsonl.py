#!/usr/bin/env python3
"""Minimal JSONL host optimizer for ``dag-ml-cli run-host-hpo`` examples.

The scheduler owns trials and scores. This adapter only proposes parameters,
receives fold feedback, and acknowledges durable checkpoint boundaries.
"""

import json
import sys


def main():
    for line in sys.stdin:
        event = json.loads(line)
        operation = event["operation"]
        if operation == "init":
            reply = {"prepared_checkpoint": None, "interrupted": []}
        elif operation == "ask":
            reply = {"params": {"n_components": event["trial_index"] + 1}}
        elif operation == "report_intermediate":
            reply = {"prune": False}
        elif operation in {"tell", "pruned", "fail", "prepare_terminal", "checkpoint"}:
            reply = {"ok": True}
        else:
            reply = {"error": f"unsupported HPO operation {operation}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
