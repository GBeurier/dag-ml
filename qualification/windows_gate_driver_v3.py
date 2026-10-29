import hashlib,importlib.metadata,json,os,pathlib,platform,runpy,sys
root=pathlib.Path(r'D:\workspace\nirs4all-release-20261029-quality\dag-ml')
wheel=pathlib.Path(r'D:\workspace\nirs4all-release-20261029-quality\wheels\dag_ml-0.3.43-cp311-abi3-win_amd64.whl')
site=root/'qualified-windows-wheel'
sys.path[:0]=[str(site),str(root)]
import dag_ml,dag_ml._dag_ml as native
pyd=pathlib.Path(native.__file__).resolve()
assert sys.platform=='win32' and platform.python_version()=='3.13.9'
assert dag_ml.version()==native.version()==importlib.metadata.version('dag-ml')=='0.3.43'
assert pyd.is_relative_to(site) and pyd.read_bytes()[:2]==b'MZ'
assert hashlib.sha256(wheel.read_bytes()).hexdigest()=='6eafd77c3dc574335cfebdb845b98d9ebbcfdb758c270b7b6fa42c9474253fdd'
observation=dict(platform=sys.platform,python=platform.python_version(),executable=sys.executable,source_commit='5d4f7be14135bd0dcf78bf55af2c7849622140e3',version=dag_ml.version(),native_path=str(pyd),native_sha256=hashlib.sha256(pyd.read_bytes()).hexdigest(),wheel=str(wheel),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),contract=json.loads(dag_ml.contract_manifest_json()))
(pathlib.Path(__file__).parent/'dag-windows-native-observation.json').write_text(json.dumps(observation,indent=2))
print('NATIVE_IDENTITY_OK',sys.platform,platform.python_version(),dag_ml.version(),str(pyd),flush=True)
gate=sys.argv[1]
targets={'training-result':'crates/dag-ml-py/tests/test_training_result.py','terminal-predict-facade':'crates/dag-ml-py/tests/test_terminal_predict_facade.py','installed-wheel-smoke':'scripts/smoke_python_bindings.py'}
if gate!='native-identity':
 target=root/targets[gate];sys.argv=[str(target),'-v'];runpy.run_path(str(target),run_name='__main__')
