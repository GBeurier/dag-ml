#!/usr/bin/env python3
import json,sys,os
SC=os.environ.get('SC','control'); LOG=os.environ.get('LOG'); STATE=os.environ.get('STATE')
def log(*a):
    if LOG: open(LOG,'a').write(' '.join(map(str,a))+'\n')
for line in sys.stdin:
    e=json.loads(line); op=e['operation']; t=e.get('trial_index'); log(op,t if t is not None else (e.get('status') or ''))
    raw=None
    if op=='init':
        r={'prepared_checkpoint':None,'interrupted':[]}
        if SC=='init-dup': raw='{"prepared_checkpoint":null,"prepared_checkpoint":null,"interrupted":[]}'
        if SC=='init-typed': r={'prepared_checkpoint':5,'interrupted':[]}
        if SC=='init-interrupted-typed': r={'prepared_checkpoint':None,'interrupted':5}
        if SC=='resume-prepared': r={'prepared_checkpoint':json.load(open(STATE)),'interrupted':[]}
        if SC=='resume-interrupted': r={'prepared_checkpoint':None,'interrupted':[{'trial_index':1,'params':{'offset':2}}]}
        if SC=='resume-interrupted-positional': r={'prepared_checkpoint':None,'interrupted':[[1,{'offset':2}]]}
        if SC=='resume-prepared-positional':
            c=json.load(open(STATE)); r={'prepared_checkpoint':[c['schema_version'],c['binding'],c['trials'],c['fingerprint']],'interrupted':[]}
    elif op=='ask':
        r={'params':{'offset':t+1}}
        if SC=='params-array': r={'params':[1]}
        if SC=='params-string': r={'params':'x'}
        if SC=='params-number': r={'params':3}
        if SC=='params-null': r={'params':None}
        if SC=='crash-at-ask1' and t==1: sys.exit(3)
        if SC=='float-exponent': raw='{"params":{"offset":%d,"probe":2e+19}}'%(t+1)
        if SC=='ask-dup': raw='{"params":{"offset":1,"offset":2}}'
        if SC=='ask-oob': raw='{"params":{"offset":1,"x":18446744073709551616}}'
        if SC=='ask-nfc': raw='{"params":{"offset":1,"\\u00e9":1,"e\\u0301":2}}'
        if SC=='ask-escape-dup': raw='{"params":{"offset":1,"o\\u0066fset":2}}'
        if SC=='extra-oob': raw='{"params":{"offset":%d},"probe":18446744073709551616}'%(t+1)
    elif op=='report_intermediate': r={'prune':False}
    elif op in('tell','pruned','fail','prepare_terminal','checkpoint'):
        r={'ok':True}
        if op=='prepare_terminal' and STATE: open(STATE,'w').write(json.dumps(e['checkpoint']))
        if SC=='tell-dup' and op=='tell': raw='{"ok":false,"ok":true}'
        if SC=='prepare-dup' and op=='prepare_terminal': raw='{"ok":false,"ok":true}'
        if SC=='checkpoint-dup' and op=='checkpoint': raw='{"ok":false,"ok":true}'
        if SC=='error-dup' and op=='tell': raw='{"ok":true,"error":"x","error":null}'
    else: r={'error':'unsupported'}
    print(raw if raw is not None else json.dumps(r),flush=True)
