"""Native identity and original short test commands; no new scientific assertions."""
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
