#!/usr/bin/env python3
"""Bounded, independent literal encoder for the adopted integrity corpus.

No production validator/encoder imports. Run only against the frozen 28/60
corpus, or use --check to prove appended literals reproducible without writes.
The legacy entries are kept as exact source byte slices, not reserialized.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / 'docs/contracts/realization_evidence.v1.schema.json'
PINS = ROOT / 'crates/dag-ml-core/tests/fixtures/realization_legacy_pins.json'
PROFILE = 'dagml-realization-evidence-v1'


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tcv(value):
    """Direct grammar transcription, independent of both owning implementations."""
    length = lambda n: struct.pack('>Q', n)
    if value is None:
        return b'N'
    if type(value) is bool:
        return b'T' if value else b'F'
    if type(value) is int:
        raw = str(value).encode('ascii')
        return b'I' + length(len(raw)) + raw
    if type(value) is str:
        raw = unicodedata.normalize('NFC', value).encode('utf-8')
        return b'S' + length(len(raw)) + raw
    if type(value) is list:
        return b'A' + length(len(value)) + b''.join(map(tcv, value))
    if type(value) is not dict:
        raise ValueError('fixture encoder accepts no floats or external objects')
    pairs = sorted((unicodedata.normalize('NFC', k), v) for k, v in value.items())
    return b'O' + length(len(pairs)) + b''.join(tcv(k) + tcv(v) for k, v in pairs)


def preimage(record):
    frame = record['record_type'] == 'frame'
    field = 'frame_digest' if frame else 'completion_digest'
    domain = b'DAGML-REALIZATION-FRAME-V1\0' if frame else b'DAGML-REALIZATION-COMPLETE-V1\0'
    return domain + b'DAGML-TCV1\0' + tcv({k: v for k, v in record.items() if k != field})


def seal(record):
    field = 'frame_digest' if record['record_type'] == 'frame' else 'completion_digest'
    record[field] = sha(preimage(record))


def owner_vectors():
    vectors = []
    for value_hex in ['0001ff', '']:
        value = dict(kind='owner_encoded', profile='owner:opaque', version='1', value_hex=value_hex)
        raw = b'DAGML-TCV1\0' + tcv(value)
        vectors.append(dict(value=value, json=dump(value), preimage_hex=raw.hex(), sha256=sha(raw)))
    return vectors


def encode(events, chunk=65536):
    raw = ''.join(dump(e) + '\n' for e in events).encode('utf-8')
    records, previous = [], None
    for at in range(0, len(raw), chunk):
        payload = raw[at:at + chunk]
        f = dict(schema_version=1, profile=PROFILE, record_type='frame',
                 capture_id='capture:one', sequence=len(records), previous_digest=previous,
                 byte_length=len(payload), payload_hex=payload.hex(), frame_digest='')
        seal(f)
        records.append(f)
        previous = f['frame_digest']
    counts = {k: 0 for k in ('scope_count', 'visited_count', 'candidate_count', 'rejected_count',
              'terminal_without_candidate_count', 'trial_count', 'new_trial_count',
              'restored_trial_count', 'resume_observation_count', 'completed_count',
              'failed_count', 'pruned_count', 'cancelled_count')}
    visits, closures = {}, bytearray(b'DAGML-REALIZATION-SCOPES-V1\0')
    for e in events:
        kind, scope = e['kind'], e['scope_id']
        if kind in ('declaration', 'scope_open'):
            counts['scope_count'] += 1
        if kind == 'visit':
            counts['visited_count'] += 1
            visits[scope, e['visit_id']] = False
        if kind in ('candidate', 'reject'):
            visits[scope, e['visit_id']] = True
            counts['candidate_count' if kind == 'candidate' else 'rejected_count'] += 1
        if kind == 'trial_visit':
            counts['trial_count'] += 1
            counts[e['origin'] + '_trial_count'] += 1
        if kind == 'trial_resume':
            counts['resume_observation_count'] += 1
        if kind == 'trial_outcome' and e['outcome']['status'] != 'running':
            counts[e['outcome']['status'] + '_count'] += 1
        if kind in ('search_close', 'scope_close'):
            scope_bytes = scope.encode('ascii')
            closures.extend(struct.pack('>Q', len(scope_bytes)) + scope_bytes)
    counts['terminal_without_candidate_count'] = sum(not v for v in visits.values())
    c = dict(schema_version=1, profile=PROFILE, record_type='completion', capture_id='capture:one',
             frame_count=len(records), last_frame_digest=previous,
             closed_scopes_digest=sha(closures), counts=counts, completion_digest='')
    seal(c)
    records.append(c)
    return records


def wire(records):
    return ''.join(dump(r) + '\n' for r in records)


def repair_tail(records, changed):
    """Repair only later chain links, leaving the isolated target fault intact."""
    previous = records[changed]['frame_digest']
    for record in records[changed + 1:]:
        if record['record_type'] == 'frame':
            record['previous_digest'] = previous
            seal(record)
            previous = record['frame_digest']
        else:
            record['last_frame_digest'] = previous
            seal(record)


def literal(name, events, chunk=65536):
    events = copy.deepcopy(events)
    records = encode(events, chunk)
    return dict(id=name, chunk_bytes=chunk, events=events, wire=wire(records),
                expected={k: records[-1][k] for k in ('capture_id', 'profile', 'schema_version', 'completion_digest')},
                preimages_hex=[preimage(r).hex() for r in records])


def entries(raw, section):
    """Raw object spans including every original internal whitespace byte."""
    start = raw.index('"' + section + '": [', raw.index('"x-conformance"'))
    at = raw.index('[', start) + 1
    decoder, spans = json.JSONDecoder(), []
    while True:
        while raw[at].isspace() or raw[at] == ',':
            at += 1
        if raw[at] == ']':
            return spans, at
        value, end = decoder.raw_decode(raw, at)
        spans.append((value, raw[at:end]))
        at = end


def build(valid):
    bases = {v['id']: v for v in valid}
    def es(name='concrete_root_node_free_source_preserved'):
        return copy.deepcopy(bases[name]['events'])
    def event(events, kind):
        return next(e for e in events if e['kind'] == kind)
    positives, negatives = [], []
    def positive(name, events, chunk=65536):
        value = literal(name, events, chunk)
        positives.append(value)
        return value
    def negative(name, events, error, refusal_at='record'):
        events = copy.deepcopy(events)
        negatives.append(dict(id=name, events=events, wire=wire(encode(events)), error=error,
                              refusal_at=refusal_at))

    base = es()
    multi = positive('integrity_multiframe', base, 1024)
    # All altered semantic fields are enclosed in independently sealed frames.
    for field, val in [('sequence', 9), ('previous_digest', 'b' * 64),
                       ('capture_id', 'capture:other'), ('byte_length', 1)]:
        records = [json.loads(line) for line in multi['wire'].splitlines()]
        records[1][field] = val
        seal(records[1])
        repair_tail(records, 1)
        negatives.append(dict(id='integrity_frame_' + field, wire=wire(records),
                              error={'sequence': 'frame_chain', 'previous_digest': 'frame_chain',
                                     'capture_id': 'capture_id', 'byte_length': 'byte_length'}[field]))
    records = encode(base)
    records[0]['frame_digest'] = '0' * 64
    repair_tail(records, 0)
    negatives.append(dict(id='integrity_frame_digest', wire=wire(records), error='frame_digest'))
    records = encode(base)
    # A single payload byte, still hex and valid byte_length, fails its checksum.
    records[0]['payload_hex'] = '00' + records[0]['payload_hex'][2:]
    negatives.append(dict(id='integrity_payload_byte', wire=wire(records), error='frame_digest'))
    for field, val, cause in [('capture_id', 'capture:other', 'capture_id'),
                              ('frame_count', 2, 'completion_chain'),
                              ('last_frame_digest', 'b' * 64, 'completion_chain'),
                              ('closed_scopes_digest', 'b' * 64, 'completion_counts'),
                              ('counts', None, 'completion_counts')]:
        records = encode(base)
        if field == 'counts':
            records[-1][field]['visited_count'] += 1
        else:
            records[-1][field] = val
        seal(records[-1])
        negatives.append(dict(id='integrity_completion_' + field, wire=wire(records), error=cause))
    records = encode(base)
    records[-1]['completion_digest'] = '0' * 64
    negatives.append(dict(id='integrity_completion_digest', wire=wire(records), error='completion_digest'))

    # Every outer string grammar site; recomputed checksum makes these single faults.
    for record_at, fields in [(0, ['capture_id', 'payload_hex', 'frame_digest', 'previous_digest']),
                              (-1, ['capture_id', 'last_frame_digest', 'closed_scopes_digest', 'completion_digest'])]:
        for field in fields:
            records = encode(base)
            records[record_at][field] = 'invalid/' if field == 'capture_id' else 'G0'
            if field not in ('frame_digest', 'completion_digest'):
                seal(records[record_at])
            negatives.append(dict(id=('integrity_outer_frame_' if record_at == 0 else 'integrity_outer_completion_') + field,
                                  wire=wire(records), error='wire_schema'))
    for field, val in [('choice_id', 'invalid/'), ('value_hex', 'GG')]:
        events = es()
        event(events, 'choice')[field] = val
        negative('integrity_event_' + field, events, 'event_schema')
    events = es()
    event(events, 'candidate')['owner_fingerprint'] = 'a' * 62
    negative('integrity_event_fingerprint', events, 'event_schema')

    # Lifecycle checks, including the three distinct trial_status sites.
    events = es('adaptive_failed')
    events = events[:next(i for i,e in enumerate(events) if e['scope_id']=='t')]
    events.append(dict(kind='scope_close', scope_id='p'))
    negative('integrity_pending_trial', events, 'pending_trial')
    events = es()[:3] + [dict(kind='scope_close', scope_id='d')]
    negative('integrity_open_children', events, 'open_children')
    events = es()
    events = [e for e in events if e['kind'] != 'candidate']
    negative('integrity_pending_visit_a_zero', events, 'pending_visit')
    events = es('adaptive_failed')
    event(events, 'trial_visit')['owner_status'] = 'completed'
    negative('integrity_trial_status_visit', events, 'trial_status')
    events = es('restored_stochastic_completed_no_new_draw')
    trial = event(events, 'trial_visit')
    resume = dict(kind='trial_resume', scope_id='p', trial_ref=dict(scope_id='p',id='trial:one'),
                  observed_status='failed', owner_evidence_hex='00', provenance=copy.deepcopy(trial['provenance']))
    events.insert(events.index(trial) + 1, resume)
    negative('integrity_trial_status_resume', events, 'trial_status')
    events = es('restored_stochastic_completed_no_new_draw')
    event(events, 'trial_outcome')['outcome']['mode'] = 'transition'
    negative('integrity_trial_status_outcome_mode', events, 'trial_status')
    events = es('restored_stochastic_completed_no_new_draw')
    event(events, 'trial_outcome')['outcome']['status'] = 'failed'
    negative('integrity_trial_status_outcome_initial', events, 'trial_status')
    events = es()
    events.insert(6, dict(kind='reject',scope_id='p',visit_id='visit:one',reason='mutex'))
    negative('integrity_visit_disposition', events, 'visit_disposition')
    events = es('derived_full_u64_unused_requested_parent')
    event(events, 'seed_declared')['requested'] = None
    negative('integrity_effective_seed_requested_null', events, 'effective_seed')
    events = es()
    close = event(events, 'search_close')
    close.update(reason='count_limit',unvisited=True,basis='owner_status',evidence=dict(availability='unknown',value_hex=None))
    negative('integrity_unvisited_evidence', events, 'unvisited_evidence')
    events = es('failed_attempts_closed_carrier_unknown_ownership')
    event(events, 'capture_provenance')['failed_attempts'][0]['owned_attempts'][0]['cleanup'] = 'confirmed'
    negative('integrity_cleanup_observation', events, 'cleanup_observation')
    events = es()
    events.insert(2, copy.deepcopy(events[1]))
    negative('integrity_provenance_carrier_event', events, 'provenance_carrier')
    events = [e for e in es() if e['kind'] != 'capture_provenance']
    negative('integrity_provenance_carrier_finish', events, 'provenance_carrier', 'finish')
    events = es('adaptive_failed')
    event(events, 'trial_outcome')['scope_id'] = 'p'
    negative('integrity_trial_outcome_nonterminal', events, 'scope_role')

    # Owner decisions and their valid neighbours; no new closure matrix.
    for name, base_name in [('transition','adaptive_completed'), ('restored','restored_stochastic_completed_no_new_draw')]:
        events = es(base_name)
        event(events, 'trial_outcome')['outcome']['failure'] = dict(code='owner:error',message='Owner diagnostic',retryable=False)
        negative('integrity_completed_failure_' + name, events, 'trial_status')
    events = es('adaptive_failed')
    event(events, 'trial_outcome')['outcome']['failure'] = None
    positive('integrity_failed_null', events)
    for reason in ['count_limit', 'owner_exhausted']:
        events = es()
        event(events, 'search_close')['reason'] = reason
        negative('integrity_enumeration_' + reason, events, 'closure_reason')
    events = es()
    event(events, 'search_close').update(reason='count_limit',basis='owner_status')
    positive('integrity_count_limit_owner_status', events)

    def seed_use(scope='p'):
        e = copy.deepcopy(event(es('explicit_stochastic_use_outside_choice'), 'seed_use'))
        e['scope_id'], e['use_id'] = scope, 'choice:a'
        return e
    for order in ['choice_first', 'use_first']:
        events = es('adaptive_failed')
        choice_at = next(i for i,e in enumerate(events) if e['kind']=='choice')
        events.insert(choice_at + (order=='choice_first'), seed_use())
        negative('integrity_choice_use_' + order, events, 'duplicate_occurrence')
    events = es('adaptive_failed')
    event(events,'effective_seed_resolved')['first_actual_use'] = dict(scope_id='s',id='choice:a')
    use = seed_use('s')
    events.insert(5,use)
    positive('integrity_choice_use_distinct_scopes', events)

    # Duplicate owner IDs survive Proposal close but release at Search close.
    events = es('adaptive_failed')
    proposal = events[5:13]
    def rename_scope(value, mapping):
        if isinstance(value,str):return mapping.get(value,value)
        if isinstance(value,list):return [rename_scope(v,mapping) for v in value]
        if isinstance(value,dict):return {k:rename_scope(v,mapping) for k,v in value.items()}
        return value
    second = rename_scope(proposal, {'p':'q','t':'u'})
    second = [e for e in second if e['kind']!='choice']
    event(second,'visit')['choice_ids'] = []
    duplicate = events[:13] + second + events[13:]
    negative('integrity_sibling_owner_ref', duplicate, 'duplicate_occurrence')
    distinct = copy.deepcopy(duplicate)
    [e for e in distinct if e['kind']=='trial_visit'][1]['owner_trial_ref_hex']='02'
    positive('integrity_sibling_distinct_owner_refs', distinct)
    separate = events[:-2] + [events[-2]]
    new_search = rename_scope(events[2:-1], {'s':'s:second','p':'q','t':'u'})
    positive('integrity_distinct_search_owner_ref', separate + new_search + [events[-1]])
    standalone = copy.deepcopy([e for e in events if e['scope_id']!='s'])
    event(standalone,'scope_open')['parent_scope_id']='d'
    # Move Search-owned seed declarations/resolutions into the standalone proposal.
    seeds = rename_scope(events[3:5], {'s':'p'})
    standalone = standalone[:3] + seeds + standalone[3:]
    event(standalone,'choice')['seed_ref']['scope_id']='p'
    standalone[-2] = dict(kind='search_close',scope_id='p',reason='count_limit',unvisited=None,
                          basis='owner_status',evidence=dict(availability='unknown',value_hex=None))
    second_standalone = rename_scope(standalone[2:-1], {'p':'q','t':'u'})
    positive('integrity_standalone_owner_ref',standalone[:-1] + second_standalone + [standalone[-1]])

    # A nested Declaration creates a genuine standalone Proposal namespace.
    # Its owner reference may repeat the still-open outer Search's trial ref.
    events = es('adaptive_failed')
    declaration = copy.deepcopy(events[0])
    declaration.update(scope_id='d:inner', parent_scope_id='p')
    inner_open = copy.deepcopy(events[5])
    inner_open.update(scope_id='p:inner', parent_scope_id='d:inner')
    inner_open['declaration_ref']['scope_id'] = 'd:inner'
    inner_visit = copy.deepcopy(events[7])
    inner_visit.update(scope_id='p:inner', visit_id='visit:inner', choice_ids=[])
    inner_trial = copy.deepcopy(events[8])
    inner_trial.update(scope_id='p:inner', trial_id='trial:inner',
                       visit_ref=dict(scope_id='p:inner', id='visit:inner'))
    terminal_open = copy.deepcopy(events[9])
    terminal_open.update(scope_id='t:inner', parent_scope_id='p:inner')
    terminal_open['declaration_ref']['scope_id'] = 'd:inner'
    outcome = copy.deepcopy(events[10])
    outcome.update(scope_id='t:inner', trial_ref=dict(scope_id='p:inner', id='trial:inner'))
    close = copy.deepcopy(events[13])
    close['scope_id'] = 'p:inner'
    nested = [declaration, inner_open, inner_visit, inner_trial, terminal_open, outcome,
              dict(kind='scope_close', scope_id='t:inner'), close,
              dict(kind='scope_close', scope_id='d:inner')]
    positive('integrity_nested_standalone_owner_ref', events[:9] + nested + events[9:])

    for value_hex in ['0001ff','']:
        events = es('adaptive_failed')
        owner = dict(kind='owner_encoded',profile='owner:opaque',version='1',value_hex=value_hex)
        event(events,'seed_declared')['requested']=copy.deepcopy(owner)
        event(events,'effective_seed_resolved')['effective']=copy.deepcopy(owner)
        positive('integrity_owner_encoded_' + ('empty' if not value_hex else 'fixed'), events)
        if value_hex:
            for field,val in [('profile','owner:other'),('version','2'),('value_hex','0001fe'),('kind','u64')]:
                changed=copy.deepcopy(events)
                effective=event(changed,'effective_seed_resolved')['effective']
                if field=='kind':
                    effective.clear();effective.update(kind='u64',value=1)
                else:effective[field]=val
                negative('integrity_owner_encoded_mismatch_' + field,changed,'seed_resolution')
            changed=copy.deepcopy(events)
            event(changed,'seed_declared')['requested']['value_hex']='0g'
            negative('integrity_owner_encoded_bad_hex',changed,'event_schema')
    # Give every new decodable wire-only mutation an independent event input.
    # Parser-corrupted payloads are explicitly enumerated in the schema instead.
    for case in negatives:
        if 'events' not in case and case['id'] not in (
                'integrity_payload_byte', 'integrity_outer_frame_payload_hex'):
            case['events'] = copy.deepcopy(base)
            case['refusal_at'] = 'wire'
    return positives, negatives


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--refresh-appended', action='store_true')
    args=parser.parse_args()
    raw=SCHEMA.read_text()
    original={k:entries(raw,k)[0] for k in ('valid','invalid')}
    pins={k:[dict(id=v['id'],raw_sha256=sha(text.encode())) for v,text in original[k][:n]]
          for k,n in [('valid',28),('invalid',60)]}
    positive,negative=build([v for v,_ in original['valid'][:28]])
    if args.check:
        if json.loads(PINS.read_text())!=pins:raise SystemExit('legacy byte pin mismatch')
        for k,new,n in [('valid',positive,28),('invalid',negative,60)]:
            if [v for v,_ in original[k][n:]]!=new:raise SystemExit('literal drift: '+k)
        if json.loads(raw)['x-owner-encoded-vectors']!=owner_vectors():
            raise SystemExit('owner seed vector drift')
        if payload_catalogues(json.loads(raw)) != (
                json.loads(raw)['x-fixture-events'], json.loads(raw)['x-fixture-payload-exceptions']):
            raise SystemExit('fixture payload catalogue drift')
    elif args.refresh_appended:
        if json.loads(PINS.read_text()) != pins:
            raise SystemExit('refuse legacy byte drift')
        for key, new, count in [('invalid', negative, 60), ('valid', positive, 28)]:
            spans, end = entries(raw, key)
            start = raw.index('"' + key + '": [', raw.index('"x-conformance"'))
            at = raw.index(spans[count-1][1], start) + len(spans[count-1][1])
            addition = ''.join(',\n      ' + json.dumps(v, ensure_ascii=True, indent=2).replace('\n','\n      ') for v in new)
            raw = raw[:at] + addition + '\n    ' + raw[end:]
        document = json.loads(raw)
        if 'x-fixture-events' in document:
            raise SystemExit('refresh requires the reviewed pre-catalogue source')
        companion, exceptions = payload_catalogues(document)
        at = raw.index('  "x-semantic-rules": [')
        raw = raw[:at] + '  "x-fixture-events": ' + json.dumps(companion, ensure_ascii=True, indent=2) + ',\n' + \
              '  "x-fixture-payload-exceptions": ' + json.dumps(exceptions, indent=2) + ',\n' + raw[at:]
        SCHEMA.write_text(raw)
    else:
        if len(original['valid'])!=28 or len(original['invalid'])!=60:
            raise SystemExit('refuse to regenerate an extended corpus')
        PINS.write_text(json.dumps(pins,indent=2)+'\n')
        # Insert immediately after the last original object. No original byte changes.
        for k,new in [('invalid',negative),('valid',positive)]:
            _,at=entries(raw,k)
            at=raw.rfind('}',0,at)+1
            addition=''.join(',\n      '+json.dumps(v,ensure_ascii=True,indent=2).replace('\n','\n      ') for v in new)
            raw=raw[:at]+addition+raw[at:]
        at=raw.index('  "x-semantic-rules": [')
        raw=raw[:at]+'  "x-owner-encoded-vectors": '+json.dumps(owner_vectors(),indent=2)+',\n'+raw[at:]
        companion, exceptions = payload_catalogues(json.loads(raw))
        at=raw.index('  "x-semantic-rules": [')
        raw=raw[:at]+'  "x-fixture-events": '+json.dumps(companion,ensure_ascii=True,indent=2)+',\n'+ \
            '  "x-fixture-payload-exceptions": '+json.dumps(exceptions,indent=2)+',\n'+raw[at:]
        SCHEMA.write_text(raw)
    print(json.dumps(dict(status='LITERAL_CHECK_PASS' if args.check else 'LITERALS_WRITTEN',
                          legacy_valid=28,legacy_invalid=60,added_valid=len(positive),added_invalid=len(negative))))


def payload_catalogues(document):
    """Explicit malformed controls; companions preserve wire-only legacy objects.

    Companions are literal annotations of the sealed legacy payloads, not new
    independent encoder claims. New decodable cases must carry their own events.
    """
    exceptions = {'separator_' + name: 'Intentionally invalid outer record separator'
                  for name in ('cr','vt','ff','fs','gs','rs','nel','ls','ps')}
    exceptions.update({
        'huge_int': 'Intentionally out-of-range 5000-digit outer integer',
        'logical_huge_integer': 'Intentionally out-of-range 5000-digit logical integer',
        'logical_negative_zero': 'Intentionally forbidden lexical integer -0',
        'logical_duplicate_key': 'Intentionally duplicated logical object member',
        'logical_nfc_key_collision': 'Intentionally NFC-colliding logical keys',
        'integrity_payload_byte': 'Intentionally corrupted payload byte for checksum refusal',
        'integrity_outer_frame_payload_hex': 'Intentionally malformed hex cannot carry logical events',
    })
    companion = {}
    for section in ('valid', 'invalid', 'parser_valid'):
        for case in document['x-conformance'][section]:
            if 'events' in case or case['id'] in exceptions:
                continue
            records = [json.loads(line) for line in case['wire'].split('\n') if line.strip()]
            payload = b''.join(bytes.fromhex(r['payload_hex']) for r in records if r['record_type']=='frame')
            companion[case['id']] = [json.loads(line) for line in payload.decode().split('\n') if line]
    return companion, exceptions


if __name__=='__main__':
    main()
