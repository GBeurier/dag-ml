#!/usr/bin/env python3
"""Sequential authorized DAG041 gates with bounded task-owned C: storage."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, signal, subprocess, time
W=Path('/home/delete/nirs4all/_worktrees/2026-10-07-dag-metric-validation')
S=Path('/mnt/c/Temp/n4a-dag041-20261008')
A=Path('/home/delete/nirs4all/_audits/2026-10-07-strict-speed-diagnostic')
PY='/home/delete/nirs4all/nirs4all/.venv/bin/python'
RTK='/home/delete/.local/bin/rtk'
SOURCE={'criteria.rs':'fe11ed0e1c503078bb20924efbaf3960955c52db8c16d3fb725969b1afe20f14','metric_provider.rs':'c4c6e9561647b430465fe2d147b997c340685ff6b8acc7fbec0333cb63324fae'}
GATES={
'metadata-root': ['cargo','metadata','--offline','--locked','--format-version','1'],
'metadata-python': ['cargo','metadata','--offline','--locked','--format-version','1','--manifest-path','crates/dag-ml-py/Cargo.toml'],
'release-metadata':[PY,'scripts/validate_release_metadata.py','--release'],
'publish-plan':[PY,'scripts/release/check_publish_plan.py'],
'abi':[PY,'scripts/validate_abi_snapshot.py'],
'contracts':[PY,'scripts/validate_contracts.py','--require-sibling','--sibling-root','/home/delete/nirs4all/dag-ml-data'],
'fmt':['cargo','fmt','--all','--check'],
'diff':['git','diff','--check'],
'maturin-cached-terminal':['maturin','build','--release','--locked','--manifest-path','crates/dag-ml-py/Cargo.toml','--out',str(S/'wheels-terminal-replay')],
 'pack-refresh':[PY,'scripts/refresh_current_training_packs.py'],
 'contracts-after-refresh':[PY,'scripts/validate_contracts.py','--require-sibling','--sibling-root','/home/delete/nirs4all/dag-ml-data'],
 'replay-contracts':[PY,'scripts/validate_training_replay_contracts.py'],
 'replay-tests':[PY,'-m','pytest','--capture=sys','-q','parity/training/tests/test_training_replay_contracts.py'],
 'wheel-metadata':[PY,'scripts/smoke_python_wheel_metadata.py',str(S/'wheels/dag_ml-0.3.41-cp311-abi3-manylinux_2_34_x86_64.whl')],
 'maturin':['maturin','build','--release','--locked','--manifest-path','crates/dag-ml-py/Cargo.toml','--out',str(S/'wheels')],
'clippy':['cargo','clippy','--offline','--locked','--workspace','--all-targets','--','-D','warnings'],
'workspace-tests':['cargo','test','--offline','--locked','--workspace'],
'core-preserve-order':['cargo','test','--offline','--locked','--manifest-path','crates/dag-ml-core/Cargo.toml','--features','serde_json/preserve_order'],
'python-default':['cargo','test','--offline','--locked','--manifest-path','crates/dag-ml-py/Cargo.toml'],
'python-preserve-order':['cargo','test','--offline','--locked','--manifest-path','crates/dag-ml-py/Cargo.toml','--features','serde_json/preserve_order'],
'python-methods':['cargo','test','--offline','--locked','--manifest-path','crates/dag-ml-py/Cargo.toml','--features','methods-optimizer'],
 'python-methods-correct-env':['cargo','test','--offline','--locked','--manifest-path','crates/dag-ml-py/Cargo.toml','--features','methods-optimizer'],
'cli-minimal':['cargo','run','--offline','--locked','-p','dag-ml-cli','--','validate-graph','examples/minimal_graph.json'],
'source-training-smoke':[PY,str(A/'dag041-source-training-smoke.py')],
 'docs':['cargo','doc','--offline','--locked','--workspace','--no-deps'],
 'error-taxonomy':[PY,'scripts/check_error_taxonomy.py'],
 'tracing-privacy':[PY,'scripts/lint_tracing_fields.py'],
'deprecations':[PY,'scripts/check_deprecations.py'],
'public-docs':[PY,'scripts/check_public_docs.py'],
'freshness-self':[PY,'scripts/check_so_freshness.py','--self-test'],
'freshness':[PY,'scripts/check_so_freshness.py'],
'facade':[PY,'-m','unittest','crates/dag-ml-py/tests/test_terminal_predict_facade.py'],
'methods-facade':[PY,'-m','unittest','crates/dag-ml-py/tests/test_methods_terminal_predict_facade.py'],
'cli-cv-refit':[PY,'scripts/smoke_cv_refit_replay_output.py',str(S/'target/debug/dag-ml-cli')],
'cli-persisted-replay':[PY,'scripts/smoke_persisted_bundle_replay.py',str(S/'target/debug/dag-ml-cli')],
'core-package-extract':['bash','scripts/test_core_package_extract.sh'],
}
MINFREE=16*1024**3

def usage():
 target=0;other=0;vanished=0
 for directory,_,files in os.walk(S):
  for name in files:
   path=Path(directory)/name
   try:size=path.stat().st_size
   except FileNotFoundError:vanished+=1;continue # Rust removes temporary objects while compiling.
   if path.is_relative_to(S/'target'):target+=size
   else:other+=size
 return {'target_bytes':target,'other_bytes':other,'free_bytes':shutil.disk_usage(S).free,'vanished_owned_temporaries':vanished}

def verify_source():
 for n,h in SOURCE.items():
  assert hashlib.sha256((W/'crates/dag-ml-core/src'/n).read_bytes()).hexdigest()==h,(n,'functional source mutated')
 freeze=A/'dag041-pre-gates-source-freeze.json'
 if freeze.exists():
  for n,record in json.loads(freeze.read_text())['changed_files'].items():
   assert hashlib.sha256((W/n).read_bytes()).hexdigest()==record['sha256'],(n,'reviewed versioned source mutated')

def main():
 p=argparse.ArgumentParser();p.add_argument('gates',nargs='+',choices=GATES);args=p.parse_args()
 assert json.loads((S/'owner.json').read_text())['owner']=='/root/strict_speed_remaining_diagnostic'
 env=os.environ.copy();env.pop('RUSTFLAGS',None)
 env.update({'CARGO_TARGET_DIR':str(S/'target'),'TMPDIR':str(S/'tmp'),'CARGO_INCREMENTAL':'0','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_NET_OFFLINE':'true','PYO3_PYTHON':PY,'DAG_ML_DATA_REPO':'/home/delete/nirs4all/dag-ml-data','N4M_LIB_PATH':'/home/delete/nirs4all/_audits/2026-10-06-debt-extension/workflow/public-methods-134-full/n4m/lib/libn4m.so.2.17.0'})
 env['N4M_LIBRARY_PATH']=env['N4M_LIB_PATH']
 env['PATH']=str(Path(PY).parent)+':'+env.get('PATH','')
 for gate in args.gates:
  verify_source();before=usage();assert before['free_bytes']>MINFREE
  d=S/'gates'/gate;d.mkdir() # never overwrite a prior attempt
  e=env.copy()
  if gate=='python-methods-correct-env':e['DAG_ML_REQUIRE_N4M_TEST']='1'
  if gate=='docs':e['RUSTDOCFLAGS']='-D warnings'
  if gate=='workspace-tests':e['RUSTFLAGS']='--cfg dag_ml_workspace_contract_fixtures'
  if gate in ('facade','methods-facade','source-training-smoke'):e['PYTHONPATH']=str(W/'crates/dag-ml-py/python')
  cmd=[RTK,'proxy',*GATES[gate]];start=time.monotonic();reason=None
  with (d/'stdout.log').open('wb') as out,(d/'stderr.log').open('wb') as err:
   (d/'command.json').write_text(json.dumps(cmd)+'\n')
   child=subprocess.Popen([PY,str(A/'dag041-child-terminal.py'),str(d)],cwd=W,env=e,stdout=out,stderr=err,start_new_session=True)
   print(json.dumps({'gate':gate,'pid':child.pid,'command':cmd,'before':before}),flush=True)
   (d/'active.json').write_text(json.dumps({'pid':child.pid,'command':cmd,'worktree':str(W)},indent=2)+'\n')
   peak=before.copy()
   while child.poll() is None:
    time.sleep(5);u=usage();peak['target_bytes']=max(peak['target_bytes'],u['target_bytes']);peak['other_bytes']=max(peak['other_bytes'],u['other_bytes']);peak['free_bytes']=min(peak['free_bytes'],u['free_bytes'])
    if u['target_bytes']>20*1024**3 or u['other_bytes']>4*1024**3 or u['free_bytes']<MINFREE:
     reason='owned-storage-budget';os.killpg(child.pid,signal.SIGTERM)
     try:child.wait(timeout=20)
     except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
     break
   rc=child.wait()
  verify_source();terminal=json.loads((d/'command-exit.json').read_text()) if (d/'command-exit.json').exists() else None;receipt={'independent_command_terminal':terminal,'gate':gate,'command':cmd,'exit_code':rc,'stop_reason':reason,'seconds':time.monotonic()-start,'before':before,'after':usage(),'observed_peak':peak,'functional_source_unchanged':True,'logs':{n:{'path':str(d/n),'bytes':(d/n).stat().st_size,'sha256':hashlib.sha256((d/n).read_bytes()).hexdigest()} for n in ('stdout.log','stderr.log')}}
  (d/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(A/f'dag041-gate-{gate}.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(json.dumps({'gate':gate,'exit_code':rc,'seconds':receipt['seconds'],'after':receipt['after']}),flush=True)
  if rc or reason:
   print((d/'stderr.log').read_text(errors='replace')[-6000:],flush=True);return rc or 1
 return 0
if __name__=='__main__':raise SystemExit(main())
