#!/usr/bin/env python3
"""Offline, unwired realization contract validator; never executes a recipe.

This gate scans every stored entry and requires zero stale artifacts in the
current coherent source. Hash consistency does not qualify native execution,
host bindings or scientific parity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import struct
import sys
import unicodedata


def _audit(event, args):
    if event == "import" and args[0].split(".")[0] in {
        "nirs4all", "dag_ml", "dag_ml_data", "n4m", "pls4all", "numpy",
        "scipy", "sklearn", "pandas", "torch", "tensorflow", "jax",
    }:
        raise RuntimeError("scientific/product import prohibited by source-only gate")
    if event in {"ctypes.dlopen", "subprocess.Popen", "socket.connect", "os.system"}:
        raise RuntimeError("runtime operation prohibited by source-only gate")


sys.addaudithook(_audit)

from jsonschema import Draft202012Validator  # noqa: E402
from referencing import Registry, Resource  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "docs/contracts/realization_evidence.v1.schema.json"
PROFILE = "dagml-realization-evidence-v1"
MAX_MEMORY = 64 * 1024 * 1024
MAX_FRAME = 1024 * 1024
IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z", re.ASCII)
HEX = re.compile(r"(?:[0-9a-f]{2})*\Z", re.ASCII)
SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
FRAME_DOMAIN = b"DAGML-REALIZATION-FRAME-V1\0"
COMPLETION_DOMAIN = b"DAGML-REALIZATION-COMPLETE-V1\0"
SCOPE_DOMAIN = b"DAGML-REALIZATION-SCOPES-V1\0"
PACKS = {
    "training_contract_conformance_pack.v1.json": 131,
    "training_replay_contract_conformance_pack.v1.json": 186,
    "conformal_robustness_conformance_pack.v1.json": 66,
    "criteria_conformance_pack.v1.json": 56,
}
VALIDATOR_SHA = "f5b24d560d8e2314c99d823a61a21d1a76527d8f283808058715f148788e70ce"


class Invalid(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def require(condition, code):
    if not condition:
        raise Invalid(code)


def strict_json(raw):
    def pairs(values):
        output, normalized = {}, set()
        for key, value in values:
            require(key not in output, "duplicate_key")
            nfc = unicodedata.normalize("NFC", key)
            require(nfc not in normalized, "nfc_key_collision")
            normalized.add(nfc)
            output[key] = value
        return output

    def constant(_):
        raise Invalid("nonfinite")

    def integer(text):
        require(text != "-0", "negative_zero")
        # Bound the token before int(), including Python's configurable digit
        # limit. A 5000-digit valid JSON integer is a typed range refusal.
        require(len(text) <= 20, "integer_range")
        try:
            value = int(text)
        except ValueError as error:
            raise Invalid("integer_range") from error
        require(-(2**63) <= value <= 2**64 - 1, "integer_range")
        return value

    def binary64(text):
        value = float(text)
        require(math.isfinite(value), "nonfinite")
        return value

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_int=integer,
                           parse_float=binary64, parse_constant=constant)
        def unicode_strings(item, depth=0):
            require(depth <= 128, "parser_limit")
            if isinstance(item, str):
                item.encode("utf-8")
            elif isinstance(item, list):
                for element in item:
                    unicode_strings(element, depth + 1)
            elif isinstance(item, dict):
                for key, element in item.items():
                    key.encode("utf-8")
                    unicode_strings(element, depth + 1)
        unicode_strings(value)
        return value
    except RecursionError as error:
        raise Invalid("parser_limit") from error
    except (json.JSONDecodeError, UnicodeError) as error:
        raise Invalid("json_syntax") from error


def tcv1(value):
    """Independent TCV1 encoding; wire fields are ASCII, with no float fields."""
    def length(n):
        return struct.pack(">Q", n)

    def encode(item):
        if item is None:
            return b"N"
        if type(item) is bool:
            return b"T" if item else b"F"
        if type(item) is int:
            require(-(2**63) <= item <= 2**64 - 1, "integer_range")
            raw = str(item).encode("ascii")
            return b"I" + length(len(raw)) + raw
        if type(item) is float:
            require(math.isfinite(item), "nonfinite")
            return b"D" + struct.pack(">d", item)
        if type(item) is str:
            raw = unicodedata.normalize("NFC", item).encode("utf-8")
            return b"S" + length(len(raw)) + raw
        if type(item) is list:
            return b"A" + length(len(item)) + b"".join(encode(v) for v in item)
        require(type(item) is dict, "canonical_type")
        pairs = [(unicodedata.normalize("NFC", k), v) for k, v in item.items()]
        require(len({k for k, _ in pairs}) == len(pairs), "nfc_key_collision")
        pairs.sort(key=lambda pair: pair[0].encode("utf-8"))
        return b"O" + length(len(pairs)) + b"".join(encode(k) + encode(v) for k, v in pairs)

    return b"DAGML-TCV1\0" + encode(value)


def preimage(record):
    frame = record["record_type"] == "frame"
    field = "frame_digest" if frame else "completion_digest"
    body = {k: v for k, v in record.items() if k != field}
    return (FRAME_DOMAIN if frame else COMPLETION_DOMAIN) + tcv1(body)



# Conservative logical admission model, not RSS. Census precedes json.loads.
# Nodes is an upper token count, not a caller-provided arbitrary event count.
def census(raw):
    quoted = escaped = nonascii = False
    depth = nodes = containers = strings = string_bytes = key_bytes = unicode_keys = token_bytes = 0
    for i, char in enumerate(raw):
        if quoted:
            if not escaped and char == '"':
                quoted = False
                string_bytes += token_bytes
                j=i+1
                while j<len(raw) and raw[j] in ' \t\r\n':j+=1
                if j<len(raw) and raw[j]==':':
                    key_bytes+=token_bytes
                    if nonascii:unicode_keys+=token_bytes
            else:
                cp=ord(char);token_bytes+=1 if cp<128 else 2 if cp<2048 else 3 if cp<65536 else 4
                nonascii |= cp>=128 or (escaped and char=='u')
                if escaped:escaped=False
                elif char=='\\':escaped=True
        elif char=='"':quoted=True;strings+=1;nodes+=1;token_bytes=0;nonascii=False
        elif char in '[{':depth+=1;containers+=1;nodes+=1;require(depth<=128,'parser_limit')
        elif char in ']}':depth-=1;require(depth>=0,'json_syntax')
        elif char in '-0123456789tfn':nodes+=1
    require(not quoted and depth==0,'json_syntax')
    slots=4*nodes+8*containers
    canonical=slots*(40+64)+(12*nodes+16*containers)*48
    strings_bound=4*string_bytes+16*strings
    normalization=unicode_keys*(4*16+16*8+16*4+4*8)+32*strings
    content=slots*(48+96)+strings_bound
    serde=(min(128,containers)+1)*content+slots*256+strings_bound
    return dict(nodes=nodes,input_characters=len(raw),reservation=4096+canonical+strings_bound+normalization+serde)

class Budget:
    def __init__(self, limit=MAX_MEMORY):
        require(type(limit) is int and 4096<=limit<=MAX_MEMORY,'memory_limit')
        self.limit,self.used,self.peak=limit,4096,4096
    def charge(self,n):
        require(n<=self.limit-self.used,'evidence_budget_exceeded')
        self.used+=n;self.peak=max(self.peak,self.used)
    def release(self,n):
        require(0<=n<=self.used,'accounting');self.used-=n

def decode(raw,budget):
    bound=census(raw)['reservation'];budget.charge(bound)
    try:return strict_json(raw)
    finally:budget.release(bound)

def record_text(raw):
    text=raw[:-1] if raw.endswith('\n') else raw
    if raw.endswith('\n') and text.endswith('\r'):text=text[:-1]
    quoted=escaped=False
    for c in text:
        if quoted:
            if escaped:escaped=False
            elif c=='\\':escaped=True
            elif c=='"':quoted=False
        elif c=='"':quoted=True
        elif c in '\r\v\f\x1c\x1d\x1e\x85\u2028\u2029':raise Invalid('record_separator')
    return text

def record_lines(text):
    at=0
    while at<len(text):
        end=text.find('\n',at);end=len(text) if end<0 else end+1
        require(end-at<=MAX_FRAME+2,'wire_size_ascii')
        yield record_text(text[at:end]);at=end

def validators(schema):
    Draft202012Validator.check_schema(schema)
    wire=Draft202012Validator(schema)
    return wire,Draft202012Validator({'$schema':schema['$schema'],'$ref':'#/$defs/event','$defs':schema['$defs']})

def nonempty_bytes(value, code):
    require(bool(value), code)


def bound_provenance(left, right):
    require(compatible(left, right), 'provenance_binding')


def checked_provenance(p):
    old=p['observation']=='restored'
    require(all((p[k] is not None)==old for k in ('original_capture_id','original_scope_id','original_id','checkpoint_ref')),'provenance')

    if old:nonempty_bytes(p['checkpoint_ref'],'provenance')

def compatible(a,b):
    return a['observation']==b['observation'] and (a['observation']=='new' or
        (a['checkpoint_ref'],a['original_capture_id'])==(b['checkpoint_ref'],b['original_capture_id']))

class State:
    def __init__(self,budget):
        self.budget=budget;self.scopes={};self.counts={k:0 for k in (
            'scope_count','visited_count','candidate_count','rejected_count','terminal_without_candidate_count',
            'trial_count','new_trial_count','restored_trial_count','resume_observation_count',
            'completed_count','failed_count','pruned_count','cancelled_count')}
        self.closures=hashlib.sha256(SCOPE_DOMAIN);self.closed=0;self.provenance_seen=False;self.incomplete=False
    def scope(self,s):
        require(s in self.scopes,'unresolved_reference');v=self.scopes[s];require(v is not None,'scope_closed');return v
    def ancestor(self,target,s):
        while s is not None:
            if s==target:return True
            s=self.scope(s)['parent']
        return False
    def trial_owner_scope(self, proposal):
        parent = self.scope(proposal)['parent']
        if parent is not None and self.scope(parent)['role'] == 'search':
            return parent
        return proposal
    def ref(self,kind,ref,scope,ancestor=False):
        target=ref['scope_id'];require(target==scope or (ancestor and self.ancestor(target,scope)),'scope_binding')
        v=self.scope(target)['facts'];key=(kind,ref['id']);require(key in v,'unresolved_reference');return v[key]
    def put(self,s,kind,key,value):
        v=self.scope(s);require((kind,key) not in v['facts'],'duplicate_occurrence')
        # Python object model differs from Rust. Capacity envelope is based on
        # actual empty/one-entry dictionary and scalar sizes, plus copied data.
        import copy
        def size(x):
            import sys
            n=sys.getsizeof(x)
            if isinstance(x,dict):n+=sum(size(k)+size(v) for k,v in x.items())
            elif isinstance(x,(tuple,list)):n+=sum(map(size,x))
            return n
        import sys
        charge=2*size((kind,key,value))+2*(sys.getsizeof({0:0})-sys.getsizeof({}))
        self.budget.charge(charge);v['lease']+=charge;v['facts'][kind,key]=copy.deepcopy(value)
    def node(self,s,node):
        v=self.scope(s);d=self.scope(v['declaration']);
        if node in d['nodes']:return node
        n=self.ref('node',dict(scope_id=s,id=node),s);return n['root']
    def seed_used(self,ref,s,ident):
        while True:
            seed=self.ref('resolution',ref,s,True)
            if seed['used']:return
            require(seed['first_actual_use']==dict(scope_id=s,id=ident),'seed_use_binding')
            seed['used']=True
            parent=seed['parent']
            if parent is None or parent['value_slot']!='effective':return
            ref=parent['seed_ref']
    def accept(self,e):
        s,k=e['scope_id'],e['kind']
        if k in ('declaration','scope_open'):
            require(s not in self.scopes,'duplicate_occurrence')
            parent=e['parent_scope_id'];role=e['scope_kind']
            if k=='declaration':
                require(role=='declaration','scope_role')
                require(parent is None or self.scope(parent)['role']=='proposal','scope_role')
                require(hashlib.sha256(e['declaration_text'].encode()).hexdigest()==e['source_digest'],'source_digest')
                require(len(set(e['node_ids']))==len(e['node_ids']),'duplicate_reference')
                declaration=s;nodes=e['node_ids'];digest=e['source_digest']
            else:
                pv=self.scope(parent)
                require((role=='search' and pv['role']=='declaration') or
                    (role=='proposal' and pv['role'] in ('search','declaration')) or
                    (role=='terminal' and pv['role']=='proposal'),'scope_role')
                declaration=parent if pv['role']=='declaration' else pv['declaration']
                require(e['declaration_ref']==dict(scope_id=declaration,source_digest=self.scope(declaration)['digest']),'foreign_declaration')
                nodes=[];digest=''
            import sys,copy
            node_lease=2*(sys.getsizeof(nodes)+sum(sys.getsizeof(n) for n in nodes))
            open_lease=sys.getsizeof(dict.fromkeys(range(11)))*2+node_lease+sum(sys.getsizeof(x) for x in (parent,declaration,digest))
            self.budget.charge(open_lease+sys.getsizeof(s)+2*(sys.getsizeof({0:0})-sys.getsizeof({})))
            nodes=copy.deepcopy(nodes)
            self.scopes[s]=dict(role=role,parent=parent,declaration=declaration,nodes=nodes,digest=digest,
                facts={},lease=0,open_lease=open_lease,children=0,pending=0)
            if parent is not None:self.scope(parent)['children']+=1
            self.counts['scope_count']+=1;return
        v=self.scope(s);role=v['role']
        if k=='capture_provenance':
            require(role=='declaration' and v['parent'] is None and not self.provenance_seen,'provenance_carrier')
            self.provenance_seen=True
            for f in e['failed_attempts']:
                if f['trusted_prefix_ref'] is not None:
                    nonempty_bytes(f['trusted_prefix_ref']['owner_ref_hex'],'owner_prefix_ref')
                for a in f['owned_attempts']:
                    require(a['ownership']!='confirmed' or bool(a['owner_ref_hex']),'ownership')
                    require(a['cleanup']!='confirmed' or a['status'] in ('completed','failed','pruned','cancelled'),'cleanup_observation')
            return
        if k=='seed_declared':
            require(role!='terminal','scope_role')
            checked_provenance(e['provenance']);self.put(s,'declared',e['seed_id'],e);return
        if k=='effective_seed_resolved':
            require(role!='terminal','scope_role')
            checked_provenance(e['provenance']);decl=self.ref('declared',e['declared_ref'],s,True)
            r=e['resolution'];requested=decl['requested']
            require((r=='explicit' and requested is not None and requested==e['effective'] and e['parent'] is None and e['policy'] is None) or
                    (r=='generated' and requested is None and e['parent'] is None and e['policy'] is not None) or
                    (r=='derived' and e['parent'] is not None and e['policy'] is not None),'seed_resolution')
            if r=='derived':
                p=e['parent'];parent=self.ref('declared' if p['value_slot']=='requested' else 'resolution',p['seed_ref'],s,True)
                require(parent['requested' if p['value_slot']=='requested' else 'effective'] is not None,'effective_seed')
            declared=e['declared_ref'];facts=self.scope(declared['scope_id'])['facts'];key=('resolved_declared',declared['id'])
            require(key not in facts,'seed_resolution_conflict')
            self.put(declared['scope_id'],'resolved_declared',declared['id'],None)
            self.put(s,'resolution',e['seed_id'],dict(**e,used=False));return
        if k in ('choice','seed_use'):
            require(role in ('proposal','search'),'scope_role');checked_provenance(e['provenance'])
            ident=e['choice_id'] if k=='choice' else e['use_id']
            require(all((kind, ident) not in v['facts'] for kind in ('choice', 'use')),
                    'duplicate_occurrence')
            if k=='choice':
                self.node(s,e['node_id'])
                if e['node_id'] not in self.scope(v['declaration'])['nodes']:
                    require(not self.ref('node',dict(scope_id=s,id=e['node_id']),s)['shared_choices'],'clone_choice_binding')
                random=e['semantics']=='stochastic'
                require(random==(e['seed_ref'] is not None),'choice_seed')
            else:random=True
            if random:
                seed=self.ref('resolution',e['seed_ref'],s,True)
                require(compatible(seed['provenance'],e['provenance']),'seed_provenance')
                if not seed['used']:self.seed_used(e['seed_ref'],s,ident)
            self.put(s,'choice' if k=='choice' else 'use',ident,e);return
        if k=='clone':
            require(role=='proposal','scope_role');root=self.node(s,e['origin_node_id'])
            require(e['clone_node_id'] not in self.scope(v['declaration'])['nodes'],'duplicate_occurrence')
            require(e['shared_choices'] or not e['choice_ids'],'clone_choice_binding')
            require(len(set(e['choice_ids']))==len(e['choice_ids']),'duplicate_reference')
            for c in e['choice_ids']:
                choice=self.ref('choice',dict(scope_id=s,id=c),s)
                require(self.node(s,choice['node_id'])==root,'clone_choice_binding')
            self.put(s,'node',e['clone_node_id'],dict(root=root if e['shared_choices'] else e['clone_node_id'],shared_choices=e['shared_choices']));return
        if k=='visit':
            require(role=='proposal','scope_role');checked_provenance(e['provenance'])
            require(len(set(e['choice_ids']))==len(e['choice_ids']),'duplicate_reference')
            for c in e['choice_ids']:
                choice=self.ref('choice',dict(scope_id=s,id=c),s)
                bound_provenance(choice['provenance'],e['provenance'])
            self.put(s,'visit',e['visit_id'],dict(disposition=None,trials=0,terminal=0,completed=False,provenance=e['provenance']))
            self.counts['visited_count']+=1;return
        if k in ('candidate','reject'):
            require(role=='proposal','scope_role');visit=self.ref('visit',dict(scope_id=s,id=e['visit_id']),s)
            require(visit['disposition'] is None,'visit_disposition')
            if k=='candidate':
                checked_provenance(e['provenance'])
                bound_provenance(visit['provenance'],e['provenance'])
                require(len(set(e['active_node_ids']))==len(e['active_node_ids']),'duplicate_reference')
                for node in e['active_node_ids']:self.node(s,node)
                self.put(s,'variant',e['owner_variant_id'],{})
            visit['disposition']=k;self.counts['candidate_count' if k=='candidate' else 'rejected_count']+=1;return
        if k=='trial_visit':
            require(role=='proposal','scope_role');checked_provenance(e['provenance'])
            require(e['origin']==e['provenance']['observation'],'provenance')
            visit=self.ref('visit',e['visit_ref'],s)
            bound_provenance(visit['provenance'],e['provenance'])
            nonempty_bytes(e['owner_trial_ref_hex'],'owner_trial_ref')
            require(e['origin']=='restored' or e['owner_status'] in (None,'running'),'trial_status')
            bound=e['candidate_ref'] is not None
            if bound:require(e['candidate_ref']==e['visit_ref'] and visit['disposition']=='candidate','trial_binding')
            self.put(self.trial_owner_scope(s),'owner_trial',e['owner_trial_ref_hex'],{})
            self.put(s,'trial',e['trial_id'],dict(visit=e['visit_ref'],bound=bound,status=None,
                initial=e['owner_status'],origin=e['origin'],provenance=e['provenance']))
            visit['trials']+=1;v['pending']+=1;self.counts['trial_count']+=1
            self.counts[e['origin']+'_trial_count']+=1;return
        if k in ('trial_bind','trial_resume'):
            require(role=='proposal','scope_role');trial=self.ref('trial',e['trial_ref'],s)
            if k=='trial_bind':
                require(trial['status'] is None and not trial['bound'],'trial_terminal')
                require(e['candidate_ref']==trial['visit'] and self.ref('visit',trial['visit'],s)['disposition']=='candidate','trial_binding');trial['bound']=True
            else:
                checked_provenance(e['provenance']);require(e['provenance']['observation']=='restored','provenance')
                bound_provenance(trial['provenance'],e['provenance'])
                require(e['observed_status']==(trial['status'] or trial['initial']),'trial_status');self.counts['resume_observation_count']+=1
            return
        if k=='trial_outcome':
            require(role=='terminal','scope_role')
            require(e['trial_ref']['scope_id']==v['parent'],'scope_binding')
            p=v['parent'];trial=self.ref('trial',e['trial_ref'],p);o=e['outcome'];status=o['status']
            require(trial['status'] is None and status!='running','trial_terminal')
            restored=trial['origin']=='restored' and trial['initial'] not in (None,'running')
            require((o['mode']=='restored_observation')==restored,'trial_status')
            require(not restored or status==trial['initial'],'trial_status')
            for field in ('stage','reason'):require((o[field]['availability']=='known')==(o[field]['value_hex'] is not None),'observation')
            for field in ('stage','reason'):
                if o[field]['availability']=='known':nonempty_bytes(o[field]['value_hex'],'observation')
            visit=self.ref('visit',trial['visit'],p)
            require(status!='completed' or trial['bound'] or visit['disposition']=='reject','trial_binding')
            require(status!='completed' or o['failure'] is None,'trial_status')
            trial['status']=status;visit['terminal']+=1;visit['completed']|=status=='completed'
            self.scope(p)['pending']-=1;self.counts[status+'_count']+=1;return
        if k=='operation_failed':
            require(e['ownership']!='confirmed' or bool(e['owner_ref_hex']),'ownership')
            self.incomplete=True;return
        require(k in ('scope_close','search_close'),'event_schema')
        search=role=='search' or (role=='proposal' and self.scope(v['parent'])['role']=='declaration')
        require((k=='search_close')==search,'closure_role');require(v['children']==0,'open_children')
        require(v['pending']==0,'pending_trial')
        if search:
            require(e['basis']!='enumeration_complete' or e['reason']=='exhausted','closure_reason')
            require(e['reason']!='exhausted' or (e['unvisited'] is False and e['basis']=='enumeration_complete' and e['evidence']['availability']=='known'),'closure_reason')
            require(e['basis']!='operation_error' or self.incomplete,'closure_reason')
            require((e['evidence']['availability']=='known')==(e['evidence']['value_hex'] is not None),'observation')
            if e['evidence']['availability']=='known':nonempty_bytes(e['evidence']['value_hex'],'observation')
            require(e['unvisited'] is None or e['evidence']['availability']=='known','unvisited_evidence')
        for (kind,_),fact in v['facts'].items():
            if kind=='resolution':require(fact['used'],'unused_effective_seed')
            if kind=='visit' and fact['disposition'] is None:
                require(fact['trials']>0 and fact['terminal']==fact['trials'] and not fact['completed'],'pending_visit')
                self.counts['terminal_without_candidate_count']+=1
        self.budget.release(v['lease']+v['open_lease']);parent=v['parent'];self.scopes[s]=None
        if parent is not None:self.scope(parent)['children']-=1
        raw=s.encode('ascii');self.closures.update(struct.pack('>Q',len(raw))+raw);self.closed+=1
    def finish(self):
        require(self.provenance_seen,'provenance_carrier')
        require(not self.incomplete,'incomplete_error')
        require(self.counts['scope_count']>0 and self.closed==self.counts['scope_count'],'empty_or_unclosed')
        require(self.counts['visited_count']==sum(self.counts[k] for k in ('candidate_count','rejected_count','terminal_without_candidate_count')),'completion_counts')

def integer_tokens(v):
    if isinstance(v,float):return False
    if isinstance(v,dict):return all(integer_tokens(x) for x in v.values())
    if isinstance(v,list):return all(integer_tokens(x) for x in v)
    return True

class Decoder:
    def __init__(self,schema,budget):
        self.wv,self.ev=validators(schema);self.budget=budget;self.state=State(budget)
        self.pending=bytearray();self.capacity=0;self.previous=None;self.sequence=0;self.capture=None;self.completion=None
    def feed(self,line):
        require(self.completion is None,'after_completion')
        require(line and line.isascii() and len(line)<=MAX_FRAME,'wire_size_ascii')
        bound=census(line)['reservation'];self.budget.charge(bound)
        try:
            r=strict_json(line);require(self.wv.is_valid(r) and integer_tokens(r),'wire_schema')
            if self.capture is None:self.capture=r['capture_id']
            require(r['capture_id']==self.capture,'capture_id')
            require(hashlib.sha256(preimage(r)).hexdigest()==r['frame_digest' if r['record_type']=='frame' else 'completion_digest'],'frame_digest' if r['record_type']=='frame' else 'completion_digest')
            if r['record_type']=='completion':
                self.state.finish();require(not self.pending and self.sequence>0,'completion_chain')
                require(r['frame_count']==self.sequence and r['last_frame_digest']==self.previous,'completion_chain')
                require(r['counts']==self.state.counts and r['closed_scopes_digest']==self.state.closures.hexdigest(),'completion_counts')
                self.completion=r;return
            require(r['sequence']==self.sequence and r['previous_digest']==self.previous,'frame_chain')
            raw=bytes.fromhex(r['payload_hex']);require(len(raw)>0 and r['byte_length']==len(raw),'byte_length')
            for byte in raw:
                if len(self.pending)==self.capacity:
                    growth=max(1024,self.capacity);self.budget.charge(growth);self.capacity+=growth
                self.pending.append(byte)
                if byte==10:
                    try:text=record_text(self.pending.decode('utf-8'))
                    except UnicodeError as error:raise Invalid('logical_utf8') from error
                    event_bound=census(text)['reservation'];self.budget.charge(event_bound)
                    try:
                        event=strict_json(text);require(self.ev.is_valid(event) and integer_tokens(event),'event_schema');self.state.accept(event)
                    finally:self.budget.release(event_bound)
                    self.pending.clear()
            self.previous=r['frame_digest'];self.sequence+=1
        finally:self.budget.release(bound)

def inspect_stream(wire,schema,memory=MAX_MEMORY):
    decoder=Decoder(schema,Budget(memory))
    for line in record_lines(wire):decoder.feed(line)
    require(decoder.completion is not None,'missing_completion')
    return decoder.completion

def decode_expected_context(raw,schema,memory=MAX_MEMORY):
    require(raw is not None and raw.strip()!='null','expected_required')
    try:value=decode(raw,Budget(memory))
    except Invalid as error:
        if error.code=='evidence_budget_exceeded':raise
        raise Invalid('expected_shape') from error
    require(type(value) is dict and set(value)=={'capture_id','profile','schema_version','completion_digest'} and
            all(type(value[k]) is str for k in ('capture_id','profile','completion_digest')) and
            type(value['schema_version']) is int,'expected_shape')
    require(IDENTIFIER.fullmatch(value['capture_id']) is not None and SHA.fullmatch(value['completion_digest']) is not None and
            value['profile']==PROFILE and value['schema_version']==1,'expected_invalid')
    return value

def verify_stream(wire,expected,schema,memory=MAX_MEMORY):
    require(expected is not None,'expected_required')
    expected=decode_expected_context(json.dumps(expected),schema,memory)
    r=inspect_stream(wire,schema,memory)
    require(all(r[k]==expected[k] for k in ('capture_id','profile','schema_version')),'expected_context')
    require(r['completion_digest']==expected['completion_digest'],'expected_completion_mismatch')
    return r

def scan_packs(root):
    stale, counts, checked = [], {}, 0
    for name, expected_count in PACKS.items():
        pack = strict_json((root / "docs/contracts" / name).read_text())
        entries = pack["artifacts"]
        require(len(entries) == expected_count, "pack_entry_count")
        counts[name] = len(entries)
        for artifact in entries:
            relative = Path(artifact["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "pack_path")
            path = root / relative
            require(path.is_file() and not path.is_symlink(), "pack_file")
            if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
                stale.append([name, artifact["path"]])
            checked += 1
    require(not stale, "unexpected_stale_set")
    require(sum(counts.values()) == 439, "pack_total")
    require(checked == 439, "pack_entries_checked")
    require(hashlib.sha256((root / "scripts/validate_contracts.py").read_bytes()).hexdigest() == VALIDATOR_SHA, "pinned_validator")
    return {"entries": checked, "counts": counts, "stale": stale, "validator_sha256": VALIDATOR_SHA}




# Literal IDs read from the named owner schemas, not inferred from URI patterns.
COMMON_SCHEMA_IDS = {'coordinator_branch_view.schema.json': ('https://github.com/GBeurier/dag-ml/schemas/coordinator_branch_view.v1.schema.json',
                                         'https://github.com/GBeurier/dag-ml-data/schemas/coordinator_branch_view.v1.schema.json'),
 'coordinator_data_plan_envelope.schema.json': ('https://github.com/GBeurier/dag-ml/schemas/coordinator_data_plan_envelope.v1.schema.json',
                                                'https://github.com/GBeurier/dag-ml-data/schemas/coordinator_data_plan_envelope.v1.schema.json'),
 'coordinator_data_plan_envelope.v2.schema.json': ('https://github.com/GBeurier/dag-ml/schemas/coordinator_data_plan_envelope.v2.schema.json',
                                                   'https://github.com/GBeurier/dag-ml-data/schemas/coordinator_data_plan_envelope.v2.schema.json'),
 'feature_fusion_selector.schema.json': ('https://github.com/GBeurier/dag-ml/schemas/feature_fusion_selector.v1.schema.json',
                                         'https://github.com/GBeurier/dag-ml-data/schemas/feature_fusion_selector.v1.schema.json'),
 'fitted_adapter_ref.schema.json': ('https://github.com/GBeurier/dag-ml/schemas/fitted_adapter_ref.v1.schema.json',
                                    'https://github.com/GBeurier/dag-ml-data/schemas/fitted_adapter_ref.v1.schema.json')}


def same_json_structure(left, right):
    """Compare every parsed JSON value and type without normalization."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same_json_structure(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same_json_structure(a, b) for a, b in zip(left, right)
        )
    return left == right


def compare_shared_schema(name, dag_schema, dmd_schema):
    """Validate both owner IDs before excluding only those root fields."""
    require(name in COMMON_SCHEMA_IDS, "dmd_shared_schema_unknown:" + name)
    require(type(dag_schema) is dict and type(dmd_schema) is dict,
            "dmd_shared_contract:" + name)
    dag_id, dmd_id = COMMON_SCHEMA_IDS[name]
    require("$id" in dag_schema, "dag_shared_id_missing:" + name)
    require("$id" in dmd_schema, "dmd_shared_id_missing:" + name)
    require(dag_schema["$id"] == dag_id, "dag_shared_id_wrong:" + name)
    require(dmd_schema["$id"] == dmd_id, "dmd_shared_id_wrong:" + name)
    # New root mappings leave caller input and every nested field untouched.
    dag_content = {key: value for key, value in dag_schema.items() if key != "$id"}
    dmd_content = {key: value for key, value in dmd_schema.items() if key != "$id"}
    require(same_json_structure(dag_content, dmd_content),
            "dmd_shared_contract:" + name)


def check_dmd_shared_schemas(dag_root, dmd_root):
    """Require the complete known common set at the two explicit owner paths."""
    require(dmd_root.is_dir() and not dmd_root.is_symlink(), "dmd_origin")
    dag_contracts = dag_root / "docs/contracts"
    dmd_contracts = dmd_root / "docs/contracts"
    common = {path.name for path in dmd_contracts.glob("*.schema.json")
              if (dag_contracts / path.name).is_file()}
    unknown = sorted(common - COMMON_SCHEMA_IDS.keys())
    missing = sorted(COMMON_SCHEMA_IDS.keys() - common)
    require(not unknown, "dmd_shared_schema_unknown:" + ",".join(unknown))
    require(not missing, "dmd_shared_schema_missing:" + ",".join(missing))
    for name in sorted(common):
        compare_shared_schema(
            name,
            strict_json((dag_contracts / name).read_text()),
            strict_json((dmd_contracts / name).read_text()),
        )
    return sorted(common)


def check_legacy_corpus_bytes(raw, pins):
    """Pin exact original fixture object slices, including all wire bytes."""
    for section, count in (('valid', 28), ('invalid', 60)):
        require(len(pins[section]) == count, 'legacy_pin_count')
        start = raw.index('"' + section + '": [', raw.index('"x-conformance"'))
        at = raw.index('[', start) + 1
        decoder = json.JSONDecoder()
        for pin in pins[section]:
            while raw[at].isspace() or raw[at] == ',':
                at += 1
            value, end = decoder.raw_decode(raw, at)
            require(value['id'] == pin['id'] and
                    hashlib.sha256(raw[at:end].encode('utf-8')).hexdigest() == pin['raw_sha256'],
                    'legacy_corpus_bytes:' + pin['id'])
            at = end


def check_fixture_event_wire_identity(schema):
    """Associate every decodable fixture with literal events, before cause checks.

    Immutable legacy wire-only cases use a separate literal event catalogue.
    LF splitting deliberately preserves U+2028/U+2029 inside quoted source text.
    Only the explicitly listed malformed parser cases may omit this comparison.
    """
    exceptions = schema['x-fixture-payload-exceptions']
    expected_exceptions = {
        'separator_' + name for name in ('cr','vt','ff','fs','gs','rs','nel','ls','ps')
    } | {'huge_int', 'logical_huge_integer', 'logical_negative_zero',
         'logical_duplicate_key', 'logical_nfc_key_collision',
         'integrity_payload_byte', 'integrity_outer_frame_payload_hex'}
    require(set(exceptions) == expected_exceptions, 'fixture_exception_set')
    companion = schema['x-fixture-events']
    seen, used_companion, used_exceptions = set(), set(), set()
    for section in ('valid', 'invalid', 'parser_valid'):
        for case in schema['x-conformance'][section]:
            ident = case['id']
            require(ident not in seen, 'fixture_duplicate_id:' + ident)
            seen.add(ident)
            if ident in exceptions:
                require(section == 'invalid' and bool(exceptions[ident]), 'fixture_exception:' + ident)
                used_exceptions.add(ident)
                continue
            events = case.get('events')
            if events is None:
                require(ident in companion, 'fixture_events_missing:' + ident)
                events = companion[ident]
                used_companion.add(ident)
            require(type(events) is list, 'fixture_events_shape:' + ident)
            try:
                records = [json.loads(line) for line in case['wire'].split('\n') if line.strip()]
                payload = b''.join(bytes.fromhex(r['payload_hex']) for r in records
                                   if r['record_type'] == 'frame')
                lines = payload.decode('utf-8').split('\n')
                require(lines[-1] == '', 'fixture_payload_termination:' + ident)
                lines = [line.removesuffix('\r') for line in lines[:-1]]
                expected = [json.dumps(e, ensure_ascii=False, separators=(',', ':')) for e in events]
                require(lines == expected, 'fixture_event_wire_identity:' + ident)
            except (ValueError, UnicodeError, KeyError) as error:
                if isinstance(error, Invalid):
                    raise
                raise Invalid('fixture_payload_decode:' + ident) from error
    require(used_companion == companion.keys() and used_exceptions == exceptions.keys(),
            'fixture_catalogue_unused_entries')


def check_shared_corpus(schema, raw):
    """Shared corpus entry point for the CLI and X1 source helper; no owner calls."""
    Draft202012Validator.check_schema(schema)
    pins = strict_json((ROOT / 'crates/dag-ml-core/tests/fixtures/realization_legacy_pins.json').read_text())
    check_legacy_corpus_bytes(raw, pins)
    check_fixture_event_wire_identity(schema)
    for vector in schema['x-owner-encoded-vectors']:
        require(strict_json(vector['json']) == vector['value'], 'owner_seed_json')
        raw_seed = tcv1(vector['value'])
        require(raw_seed.hex() == vector['preimage_hex'] and
                hashlib.sha256(raw_seed).hexdigest() == vector['sha256'], 'owner_seed_literal')
    results=[]
    for case in schema['x-conformance']['valid']:
        result=verify_stream(case['wire'],case['expected'],schema)
        require(result['completion_digest']==case['expected']['completion_digest'],'literal_digest')
        records=list(record_lines(case['wire']))
        require(len(records)==len(case['preimages_hex']),'literal_count')
        for record,expected in zip(records,case['preimages_hex']):require(preimage(strict_json(record)).hex()==expected,'literal_preimage')
        results.append(case['id'])
    for case in schema['x-conformance']['parser_valid']:
        verify_stream(case['wire'],case['expected'],schema)
    for case in schema['x-conformance']['invalid']:
        if case.get('structural_pass'):inspect_stream(case['wire'],schema)
        try:
            if 'expected' in case:verify_stream(case['wire'],case['expected'],schema)
            else:inspect_stream(case['wire'],schema,case.get('memory_bytes',MAX_MEMORY))
        except Invalid as error:require(error.code==case['error'],'wrong_cause:'+case['id']+':'+error.code)
        else:raise Invalid('negative_accepted:'+case['id'])
    return dict(valid=len(results), parser_valid=len(schema['x-conformance']['parser_valid']),
                invalid=len(schema['x-conformance']['invalid']), legacy_valid=28, legacy_invalid=60)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dmd-root',type=Path,required=True)
    args=parser.parse_args()
    raw=(ROOT/SCHEMA).read_text()
    schema=strict_json(raw)
    corpus=check_shared_corpus(schema, raw)
    # This root is explicit and required; never substitute a fake sibling or
    # execute SDK/scientific code. Common contract bytes remain unchanged.
    common=check_dmd_shared_schemas(ROOT,args.dmd_root)
    print(json.dumps(dict(status='STATIC_ONLY_PASS',**corpus,packs=scan_packs(ROOT),dmd_root=str(args.dmd_root.resolve()),dmd_shared_schemas=common,jsonschema=importlib.metadata.version('jsonschema'),requirements_pin='4.25.1; executed version disclosed separately',rust_executed=False,owner_wiring=False,runtime_qualified=False),indent=2))

if __name__=='__main__':
    try:main()
    except Invalid as error:
        print(json.dumps(dict(status='refused',code=error.code)),file=sys.stderr)
        raise SystemExit(1)
