#!/usr/bin/env python3
"""Probe operator: real example build_result/lifecycle; raw byte injection at emission only."""
import importlib.util, json, os, sys
from pathlib import Path
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[5] / "examples/adapters/python_process_controller.py"
spec = importlib.util.spec_from_file_location("ppc", ROOT)
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
SC = os.environ.get("SC", "clean"); LOG = os.environ["PROBELOG"]; ONCE = os.environ.get("ONCE") == "1"
MARK = os.environ["PROBELOG"] + ".fired"
def log(*a):
    with open(LOG, "a") as f: f.write(" ".join(map(str, a)) + "\n")
M = '"metrics": {'
METRIC = {
 "dup": '"strict_probe":1,"strict_probe":2,',
 "oob_pos": '"strict_probe":18446744073709551616,',
 "oob_neg": '"strict_probe":-9223372036854775809,',
 "opus_combined": '"a":1,"a":18446744073709551616,',
 "escaped_dup": '"strict_probe":1,"strict\\u005fprobe":2,',
 "nfc_dup": '"\\u00e9":1,"e\\u0301":2,',
 "legal_u64max": '"strict_probe":18446744073709551615,',
 "legal_i64min": '"strict_probe":-9223372036854775808,',
 "legal_exp": '"strict_probe":1.8446744073709552e19,',
 "huge_exp": '"strict_probe":1e400,',
 "metric_str": '"strict_probe":"x",',
}
def fired():
    if not ONCE: return False
    return os.path.exists(MARK)
def fire():
    if ONCE: open(MARK, "w").close()
def mutate_result(text):
    if fired(): return text.encode()
    if SC in METRIC:
        assert M in text; fire(); return text.replace(M, M + METRIC[SC], 1).encode()
    if SC == "nested_dup":
        fire(); return text.replace('"lineage": {', '"lineage": {"seed":1,', 1).encode()
    if SC == "top_dup":
        fire(); return text.replace('{', '{"artifacts":[],', 1).encode()
    if SC == "outputs_map_dup":
        d = json.loads(text)
        if d["outputs"]:
            k, v = next(iter(d["outputs"].items())); fire()
            return text.replace('"outputs": {', '"outputs": {' + json.dumps(k) + ':' + json.dumps(v) + ',', 1).encode()
        return text.encode()
    if SC == "schema_null":
        return text.replace('{', '{"schema_version":null,', 1).encode()
    if SC == "unknown_field":
        return text.replace('{', '{"strict_extra":1,', 1).encode()
    if SC == "trailing":
        fire(); return text.encode() + b" {}"
    if SC == "invalid_utf8":
        fire(); return text.encode().replace(b'"adapter_smoke"', b'"adapter\xffsmoke"', 1)
    if SC == "null_doc":
        fire(); return b"null"
    return text.encode()
def emit(raw):
    capture = os.environ.get("PROBERAW")
    if capture:
        with open(capture, "ab") as stream:
            stream.write(raw + b"\n")
    sys.stdout.buffer.write(raw + b"\n"); sys.stdout.buffer.flush()
def result_text(task):
    return json.dumps(base.build_result(task), sort_keys=False)
def jsonl():
    log("spawn", os.getpid())
    for line in sys.stdin:
        payload = json.loads(line)
        kind = payload.get("type") if isinstance(payload, dict) else None
        log("recv", kind, "framed" if kind else "BARE")
        if kind == "init":
            if SC == "ack_dup": emit(b'{"type":"ack","schema_version":1,"status":"initialized","status":"initialized"}')
            elif SC == "ack_type_dup": emit(b'{"type":"ack","type":"ack","schema_version":1,"status":"initialized"}')
            elif SC == "ack_extra": emit(b'{"type":"ack","schema_version":1,"status":"initialized","runtime":{"execution_host":"r","dlls":{"base":"/x"}}}')
            elif SC == "ack_extra_dup": emit(b'{"type":"ack","schema_version":1,"status":"initialized","runtime":{"a":1,"a":2}}')
            elif SC == "ack_extra_oob": emit(b'{"type":"ack","schema_version":1,"status":"initialized","runtime":18446744073709551616}')
            elif SC == "ack_positional": emit(b'["ack",1,"initialized"]')
            elif SC == "ack_null_version": emit(b'{"type":"ack","schema_version":null,"status":"initialized"}')
            else: emit(b'{"type":"ack","schema_version":1,"status":"initialized"}')
        elif kind == "task":
            if SC == "error_retry_clean" and not fired():
                fire(); emit(b'{"type":"error","schema_version":1,"error":{"code":"c","message":"m","retryable":true}}'); continue
            if SC == "error_retry_dup" and not fired():
                fire(); emit(b'{"type":"error","schema_version":1,"error":{"code":"c","code":"d","message":"m","retryable":true}}'); continue
            if SC == "error_retry_positional" and not fired():
                fire(); emit(b'{"type":"error","schema_version":1,"error":["c","m",true]}'); continue
            if SC == "error_retry_oob" and not fired():
                fire(); emit(b'{"type":"error","schema_version":1,"error":{"code":"c","message":"m","retryable":true,"n":18446744073709551616}}'); continue
            if SC == "eof_once" and not fired():
                fire(); sys.exit(0)
            body = mutate_result(result_text(payload["task"]))
            if SC == "frame_outer_dup":
                emit(b'{"type":"result","schema_version":1,"schema_version":1,"result":' + body + b'}')
            elif SC == "frame_positional":
                emit(b'["result",1,' + body + b']')
            elif SC == "frame_result_extra":
                emit(b'{"type":"result","schema_version":1,"note":"x","result":' + body + b'}')
            else:
                emit(b'{"type":"result","schema_version":1,"result":' + body + b'}')
        elif kind == "close":
            emit(b'{"type":"ack","schema_version":1,"status":"closed"}'); log("close"); return
def main():
    a = sys.argv[1:]
    if a[:1] == ["--describe"]:
        d = base.adapter_description("opus-probe")
        if SC == "no_control_frames":
            d["capabilities"] = [c for c in d["capabilities"] if c != "control_frames_v1"]
        print(json.dumps(d), flush=True); return
    if a[:1] == ["--jsonl"]:
        jsonl(); return
    log("oneshot", os.getpid())
    task = json.load(sys.stdin)
    raw = mutate_result(result_text(task))
    if SC == "positional_doc":
        res = base.build_result(task); fields = ["schema_version","node_id","outputs","predictions","classification_probabilities","observation_predictions","aggregated_predictions","explanations","shape_deltas","artifacts","artifact_handles","consumed_data_views","fit_influence_diagnostics","regression_targets","lineage"]
        defaults = {"outputs":{}, "artifact_handles":{}, "consumed_data_views":{}}
        raw = json.dumps([res.get(f, defaults.get(f, [] if f not in ("schema_version",) else None)) for f in fields]).encode()
    emit(raw)
main()
