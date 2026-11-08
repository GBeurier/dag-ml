#!/usr/bin/env python3
"""Bounded X1 source helper: shared literal corpus and byte-preservation controls.

Pure Python only. No native loading, subprocesses, recipe execution or network.
The separate CLI additionally requires the actual DMD root and frozen pack scan.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = load('integrity_validator', 'scripts/validate_realization_evidence.py')
encoder = load('independent_encoder', 'scripts/extend_realization_integrity_corpus.py')


class IntegrityCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = (ROOT / validator.SCHEMA).read_text()
        cls.schema = validator.strict_json(cls.raw)
        cls.pins = json.loads(encoder.PINS.read_text())

    def test_x1_runs_the_same_complete_corpus(self):
        result = validator.check_shared_corpus(self.schema, self.raw)
        self.assertEqual(result, dict(valid=38, invalid=109, parser_valid=3,
                                      legacy_valid=28, legacy_invalid=60))

    def test_pack_scan_requires_zero_stale_entries(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contracts = root / 'docs/contracts'
            contracts.mkdir(parents=True)
            scripts = root / 'scripts'
            scripts.mkdir()
            (scripts / 'validate_contracts.py').write_bytes(
                (ROOT / 'scripts/validate_contracts.py').read_bytes())
            payload = b'synthetic owner artifact\n'
            (root / 'artifact.txt').write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            packs = {}
            for name, count in validator.PACKS.items():
                packs[name] = {'artifacts': [
                    {'path': 'artifact.txt', 'sha256': digest} for _ in range(count)]}
                (contracts / name).write_text(json.dumps(packs[name]), encoding='utf-8')
            result = validator.scan_packs(root)
            self.assertEqual(result['entries'], 439)
            self.assertEqual(result['counts'], validator.PACKS)
            self.assertEqual(result['stale'], [])
            self.assertEqual(result['validator_sha256'], validator.VALIDATOR_SHA)
            name = next(reversed(validator.PACKS))
            packs[name]['artifacts'][-1]['sha256'] = '0' * 64
            (contracts / name).write_text(json.dumps(packs[name]), encoding='utf-8')
            with self.assertRaises(validator.Invalid) as error:
                validator.scan_packs(root)
            self.assertEqual(error.exception.code, 'unexpected_stale_set')

    def test_independent_literal_generation_and_legacy_bytes(self):
        validator.check_legacy_corpus_bytes(self.raw, self.pins)
        positive, negative = encoder.build(self.schema['x-conformance']['valid'][:28])
        self.assertEqual(positive, self.schema['x-conformance']['valid'][28:])
        self.assertEqual(negative, self.schema['x-conformance']['invalid'][60:])
        # Existing fixed vectors cross-check this independent encoder before use.
        for value in self.schema['x-conformance']['valid'][:28]:
            regenerated = encoder.literal(value['id'], value['events'], value['chunk_bytes'])
            for field in ('wire', 'preimages_hex', 'expected'):
                self.assertEqual(regenerated[field], value[field], value['id'] + ':' + field)

    def test_event_wire_gate_catches_review_reproduced_aliases(self):
        validator.check_fixture_event_wire_identity(self.schema)
        for ident in ('integrity_sibling_owner_ref', 'integrity_distinct_search_owner_ref'):
            bad = copy.deepcopy(self.schema)
            case = next(c for section in ('valid','invalid') for c in bad['x-conformance'][section]
                        if c['id'] == ident)
            next(e for e in case['events'] if e['kind']=='scope_open' and e['scope_id']=='p')['parent_scope_id']='d'
            with self.assertRaises(validator.Invalid) as error:
                validator.check_fixture_event_wire_identity(bad)
            self.assertEqual(error.exception.code, 'fixture_event_wire_identity:' + ident)

    def test_event_paths_accept_all_positives_and_pin_nested_standalone(self):
        for case in self.schema['x-conformance']['valid']:
            with self.subTest(case=case['id']):
                state = validator.State(validator.Budget())
                for event in case['events']:
                    state.accept(event)
                state.finish()
        nested = next(c for c in self.schema['x-conformance']['valid']
                      if c['id']=='integrity_nested_standalone_owner_ref')
        trials = [e for e in nested['events'] if e['kind']=='trial_visit']
        self.assertEqual(trials[0]['owner_trial_ref_hex'], trials[1]['owner_trial_ref_hex'])
        self.assertNotEqual(trials[0]['scope_id'], trials[1]['scope_id'])
        sibling = next(c for c in self.schema['x-conformance']['invalid'] if c['id']=='integrity_sibling_owner_ref')
        state = validator.State(validator.Budget())
        with self.assertRaises(validator.Invalid) as error:
            for event in sibling['events']:
                state.accept(event)
        self.assertEqual(error.exception.code, 'duplicate_occurrence')

    def test_legacy_preservation_check_catches_a_wire_byte_change(self):
        changed = self.raw.replace('capture:one', 'capture:two', 1)
        with self.assertRaises(validator.Invalid) as result:
            validator.check_legacy_corpus_bytes(changed, self.pins)
        self.assertTrue(result.exception.code.startswith('legacy_corpus_bytes:'))

    def test_outer_grammar_precedes_semantics_in_both_record_arms(self):
        cases = self.schema['x-conformance']['invalid']
        selected = [c for c in cases if c['id'].startswith('integrity_outer_')]
        self.assertEqual(len(selected), 8)
        for case in selected:
            with self.subTest(case=case['id']):
                with self.assertRaises(validator.Invalid) as result:
                    validator.inspect_stream(case['wire'], self.schema)
                self.assertEqual(result.exception.code, 'wire_schema')

    def test_literal_preimages_and_sha_are_not_self_generated_by_rust(self):
        for value in self.schema['x-conformance']['valid'][28:]:
            for record, expected in zip(value['wire'].splitlines(), value['preimages_hex']):
                r = json.loads(record)
                self.assertEqual(validator.preimage(r).hex(), expected)
                field = 'frame_digest' if r['record_type'] == 'frame' else 'completion_digest'
                self.assertEqual(encoder.sha(bytes.fromhex(expected)), r[field])

    def test_dmd_drift_comparison_does_not_mutate_inputs(self):
        name = next(iter(validator.COMMON_SCHEMA_IDS))
        dag = json.loads((ROOT / 'docs/contracts' / name).read_text())
        dmd = copy.deepcopy(dag)
        dmd['$id'] = validator.COMMON_SCHEMA_IDS[name][1]
        before = copy.deepcopy(dmd)
        validator.compare_shared_schema(name, dag, dmd)
        self.assertEqual(dmd, before)
        dmd['additionalProperties'] = not dmd.get('additionalProperties', False)
        with self.assertRaises(validator.Invalid):
            validator.compare_shared_schema(name, dag, dmd)




if __name__ == '__main__':
    unittest.main(verbosity=1)
