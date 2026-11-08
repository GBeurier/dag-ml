#!/usr/bin/env python3
"""Plan-review fixture: verbatim --describe bytes; real example lifecycle for tasks."""
import importlib.util, json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("ppc", Path(__file__).resolve().parents[5] / "examples/adapters/python_process_controller.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
LOG = os.environ["PROBELOG"]
def log(*a):
    with open(LOG, "a") as f: f.write(" ".join(map(str, a)) + "\n")
def main():
    a = sys.argv[1:]
    if a[:1] == ["--describe"]:
        log("describe", os.getpid())
        sys.stdout.buffer.write(open(os.environ["DESCFILE"], "rb").read()); sys.stdout.flush()
        err = os.environ.get("DERR")
        if err: sys.stderr.write(err + "\n")
        sys.exit(int(os.environ.get("DEXIT", "0")))
    if os.environ.get("DESC_HPO_DELEGATE") == "1":
        log("oneshot", os.getpid())
        target = Path(__file__).resolve().parents[1] / "hpo_ingress" / "op.py"
        os.execv(sys.executable, [sys.executable, str(target)] + a)
    if a[:1] == ["--jsonl"]:
        log("spawn", os.getpid())
        for line in sys.stdin:
            f = json.loads(line); log("recv", f.get("type"))
            if not base.handle_control_frame(f): log("close"); break
        return
    log("oneshot", os.getpid()); base.emit_result(json.load(sys.stdin))
main()
