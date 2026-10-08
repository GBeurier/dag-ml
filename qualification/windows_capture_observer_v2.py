"""Capture four short, genuine native Windows commands and immutable evidence."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path('/mnt/d/nirs4all-release-qualification-20261007/dag041-windows-afc610cc/source-worktree')
WINROOT = r'D:\nirs4all-release-qualification-20261007\dag041-windows-afc610cc\source-worktree'
BASE = ROOT.parent
PYEXE = '/mnt/d/nirs4all-release-qualification-20261007/strict-b345054ef9584759a3c1121e3658056b/application/resources/backend/python-runtime/python/python.exe'
SOURCE = 'afc610ccbad8a3030171a1867d57afaaedffafe7'
SHARED = Path('/mnt/c/Temp/n4a-core045-20261008/audit')
HANDOFF = Path('/home/delete/nirs4all/_audits/2026-10-07-strict-speed-diagnostic/dag041-windows-policy-scope-handoff-v1.json')

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

q = load_module('qualification', SHARED / 'shared-verifier-frozen-v4/scripts/verify_local_qualification.py')
collector = load_module('collector', SHARED / 'shared-collector-frozen-v2/collect_local_qualification.py')

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def descriptor(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': q.digest(path)}

DRIVER = r'''"""Native identity and original short test commands; no new scientific assertions."""
import hashlib, importlib.metadata, json, os, pathlib, platform, runpy, sys
gate, source, wheel, observer = sys.argv[1:5]
root = pathlib.Path(source).resolve()
package = root / 'qualified-windows-wheel'
sys.path[:0] = [str(package), str(root)]
import dag_ml, dag_ml._dag_ml as native
pyd = pathlib.Path(native.__file__).resolve()
assert sys.platform == 'win32'
assert platform.python_version() == '3.11.13'
assert dag_ml.version() == native.version() == importlib.metadata.version('dag-ml') == '0.3.41'
assert pyd.is_relative_to(package) and pyd.read_bytes()[:2] == b'MZ'
assert hashlib.sha256(pyd.read_bytes()).hexdigest() == '7ed65c6d2b11bac3e816f2d535a8d292410991cc405074b643b55de2f2fe525f'
def artifact(name, version, origin, file):
    path = pathlib.Path(file).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(name=name, version=version, origin=origin, sha256=digest, bytes=path.stat().st_size, artifact_path=str(path))
sources = [artifact('dag-ml-wheel', '0.3.41', 'GHA37704714055:artifact11519496014', wheel),
           artifact('dag-ml-native', '0.3.41', 'original-wheel-extraction', pyd),
           artifact('capture-observer', '2', 'owned-local-source', observer)]
dependencies = [artifact('python', platform.python_version(), 'packaged-windows-runtime', sys.executable),
                artifact('python311.dll', platform.python_version(), 'packaged-windows-runtime', pathlib.Path(sys.executable).parent / 'python311.dll')]
assert sources[0]['sha256'] == '8d6c71afe850c740809107e3500af78032d6ec96d6048498cf24a7d97b013a67'
print('N4A_NATIVE_OBSERVATION=' + json.dumps(dict(host=platform.node(), platform=sys.platform, python=platform.python_version(), executable=sys.executable, native_contract=json.loads(dag_ml.contract_manifest_json()), provenance=dict(source_artifacts=sources, dependency_origins=dependencies)), sort_keys=True), flush=True)
targets = {'training-result': 'crates/dag-ml-py/tests/test_training_result.py',
           'terminal-predict-facade': 'crates/dag-ml-py/tests/test_terminal_predict_facade.py',
           'installed-wheel-smoke': 'scripts/smoke_python_bindings.py'}
if gate != 'native-identity':
    target = root / targets[gate]
    sys.argv = [str(target), '-v']
    runpy.run_path(str(target), run_name='__main__')
'''

assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == SOURCE
qualification = ROOT / 'qualification'
qualification.mkdir(exist_ok=False)
driver = qualification / 'windows_gate_driver.py'
driver.write_text(DRIVER, encoding='utf8')
wheel = BASE / 'wheel/dag_ml-0.3.41-cp311-abi3-win_amd64.whl'
package = ROOT / 'qualified-windows-wheel'
package.mkdir(exist_ok=False)
assert q.digest(wheel) == '8d6c71afe850c740809107e3500af78032d6ec96d6048498cf24a7d97b013a67'
with zipfile.ZipFile(wheel) as archive:
    for item in archive.infolist():
        target = package / item.filename
        assert target.resolve().is_relative_to(package.resolve())
    archive.extractall(package)

def windows_path(path):
    value = str(path)
    for prefix, drive in [('/mnt/d/', 'D:\\'), ('/mnt/c/', 'C:\\')]:
        if value.startswith(prefix):
            return drive + value[len(prefix):].replace('/', '\\')
    raise ValueError('unmounted Windows artifact')

def artifact(name, version, origin, path):
    return dict(name=name, version=version, origin=origin, sha256=q.digest(path), bytes=path.stat().st_size, artifact_path=windows_path(path))

requirements = {
    'source_artifacts': [artifact('dag-ml-wheel', '0.3.41', 'GHA37704714055:artifact11519496014', wheel), artifact('dag-ml-native', '0.3.41', 'original-wheel-extraction', package / 'dag_ml/_dag_ml.pyd'), artifact('capture-observer', '2', 'owned-local-source', Path(__file__).resolve())],
    'dependency_origins': [artifact('python', '3.11.13', 'packaged-windows-runtime', Path(PYEXE)), artifact('python311.dll', '3.11.13', 'packaged-windows-runtime', Path(PYEXE).parent / 'python311.dll')],
}
policy = {'schema': q.SCHEMA, 'project': 'dag', 'proposal_incomplete_not_for_publication': True, 'gates': []}
commands = {}
for original in json.loads(HANDOFF.read_text())['gates']:
    gate = {k: v for k, v in original.items() if k not in {'historical_reference_command_only', 'commands_must_be_bound_to_actual_fresh_driver_not_old_path'}}
    gate['input_paths'] = [*gate['input_paths'], 'qualification/windows_gate_driver.py']
    cmd = [PYEXE, '-B', WINROOT + r'\qualification\windows_gate_driver.py', gate['id'], WINROOT, windows_path(wheel), windows_path(Path(__file__).resolve())]
    gate['commands'] = {'windows': q.canonical_command(cmd, str(ROOT), gate, 'windows')}
    gate['provenance_requirements'] = requirements
    policy['gates'].append(gate)
    commands[gate['id']] = cmd
write(qualification / 'policy.json', policy)
subprocess.run(['git', '-c', 'core.filemode=false', 'add', 'qualification/policy.json', 'qualification/windows_gate_driver.py'], cwd=ROOT, check=True)
env = os.environ.copy()
forward = {'TMP': windows_path(BASE / 'tmp'), 'TEMP': windows_path(BASE / 'tmp'), 'PYTHONUTF8': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
env.update(forward)
env['WSLENV'] = env.get('WSLENV', '') + ':' + ':'.join(key + '/w' for key in forward)
runs = []
for gate in policy['gates']:
    name = gate['id']
    prefix = 'qualification/windows-v2-' + name
    pre_path, raw_path, terminal_path, log_path = [ROOT / (prefix + suffix) for suffix in ['.PRE.json', '.raw.json', '.terminal.json', '.log']]
    fingerprints = q.gate_inputs(q.tracked_inputs(ROOT), gate)
    write(pre_path, {'mode': 'captured', 'source_sha': SOURCE, 'captured_at': now(), 'input_fingerprints': fingerprints})
    started = now()
    producer_started = now()
    with log_path.open('xb') as stream:
        result = subprocess.run(commands[name], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
    producer_finished = now()
    log = log_path.read_text(encoding='utf8')
    matches = re.findall(r'^N4A_NATIVE_OBSERVATION=(.+)$', log, flags=re.M)
    assert len(matches) == 1, 'missing actual native identity'
    observation = json.loads(matches[0])
    assert result.returncode == 0, f'{name} failed; actual log retained'
    assert observation['provenance'] == requirements, 'actual native cohort differs'
    assert q.gate_inputs(q.tracked_inputs(ROOT), gate) == fingerprints, 'source changed during native process'
    passed = 0
    if gate['minimum_passed']:
        match = re.search(r'Ran (\d+) tests? in .*?\n\s*OK\s*$', log)
        assert match and 'skipped=' not in log, 'actual unit-test count missing'
        passed = int(match.group(1))
        assert passed == gate['minimum_passed']
    identity = dict(id=name, host=observation['host'], command=commands[name], cwd=str(ROOT), environment=forward, exit_code=result.returncode, input_fingerprints=fingerprints)
    raw = dict(schema=q.SCHEMA, **identity, started_at=producer_started, finished_at=producer_finished, input_evidence=descriptor(pre_path), proof={'source_sha': SOURCE, 'success': True, 'diagnostic_only': False}, summary={'passed': passed, 'failed': 0, 'skipped': 0}, skips=[], facts={'platform': observation['platform'], 'native_contract': observation['native_contract']}, provenance=observation['provenance'])
    write(raw_path, raw)
    terminal = dict(**identity, started_at=started, finished_at=now(), os_family='windows', source_sha=SOURCE, execution={'location': 'local', 'github_actions': False, 'performance_policy': 'strict'}, log=descriptor(log_path), pre_input_sha256=q.digest(pre_path), producer_report_sha256=q.digest(raw_path), tools={'python': observation['python']})
    write(terminal_path, terminal)
    runs.append(collector.finalize(q, ROOT, 'dag', terminal_path, raw_path, pre_path, prefix + '.final', {'D:': Path('/mnt/d'), 'C:': Path('/mnt/c')}))
    print(json.dumps({'gate': name, 'actual_native_windows_exit': result.returncode, 'passed': passed, 'collector': runs[-1]}), flush=True)
write(qualification / 'windows-v2-consolidation.json', {'source_origin': SOURCE, 'source_origin_tree': '7e77957b89fae7097886d310e71e249dc5eb9c45', 'captured_at': now(), 'runs': runs, 'policy': descriptor(qualification / 'policy.json'), 'driver': descriptor(driver), 'public_qualification': False, 'scope': 'four short Windows commands only; no full E2E rebuild', 'all_four_actual_native_windows_passed': len(runs) == 4})
