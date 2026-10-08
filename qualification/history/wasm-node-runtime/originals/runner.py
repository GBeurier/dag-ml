#!/usr/bin/env python3
from pathlib import Path
import argparse,datetime,hashlib,json,os,shutil,signal,subprocess,time
W=Path('/home/delete/nirs4all/_worktrees/2026-10-07-dag-metric-validation');S=Path('/mnt/c/Temp/n4a-dag041-20261008');A=Path('/home/delete/nirs4all/_audits/2026-10-07-strict-speed-diagnostic')
PY='/home/delete/nirs4all/nirs4all/.venv/bin/python';NODE='/home/delete/.nvm/versions/node/v22.21.1/bin/node';PACK='/home/delete/.cargo/bin/wasm-pack'
STAGES={
'wasm-node041-build':[PACK,'build','crates/dag-ml-wasm','--target','nodejs','--out-dir',str(S/'wasm-node041'),'--release','--mode','no-install','--offline','--locked'],
'wasm-web041-build':[PACK,'build','crates/dag-ml-wasm','--target','web','--out-dir',str(S/'wasm-web041'),'--release','--mode','no-install','--offline','--locked'],
'wasm-node-runtime':[NODE,'scripts/smoke_wasm_bindings.cjs',str(S/'wasm-node041')],
'wasm-browser-worker-hpo-resume':[NODE,'scripts/smoke_wasm_browser_hpo.mjs',str(S/'wasm-web041')],
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
 f=json.loads((A/'dag041-ci-source-freeze-v2.json').read_text())
 for n,r in f['all_source_files'].items():assert sha(W/n)==r['sha256'],n
 return {'tree':f['actual_staged_git_tree'],'files':{n:r['sha256'] for n,r in f['all_source_files'].items()}}
def usage():
 u={'target_bytes':0,'other_bytes':0,'free_bytes':shutil.disk_usage(S).free}
 for d,_,fs in os.walk(S):
  for n in fs:
   p=Path(d)/n
   try:z=p.stat().st_size
   except FileNotFoundError:continue
   u['target_bytes' if p.is_relative_to(S/'target') else 'other_bytes']+=z
 return u
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument('stages',nargs='+',choices=STAGES);a=p.parse_args()
 assert json.loads((S/'owner.json').read_text())['owner']=='/root/strict_speed_remaining_diagnostic'
 env=os.environ.copy();env.pop('RUSTFLAGS',None)
 settings={'CARGO_TARGET_DIR':str(S/'target'),'TMPDIR':str(S/'tmp'),'CARGO_INCREMENTAL':'0','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_NET_OFFLINE':'true','WASM_PACK_CACHE':str(S/'wasm-tool-cache'),'CHROME_BIN':'/home/delete/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome','DAG_ML_DATA_REPO':'/home/delete/nirs4all/dag-ml-data'}
 env.update(settings);env['PATH']=str(Path(NODE).parent)+':/home/delete/.cargo/bin:'+env['PATH']
 for stage in a.stages:
  inputs=verify();before=usage();assert before['free_bytes']>=16*1024**3
  d=S/'gates'/stage;d.mkdir();cmd=['/home/delete/.local/bin/rtk','proxy',*STAGES[stage]]
  started=now();begin=time.monotonic();(d/'command.json').write_text(json.dumps(cmd)+'\n')
  capture={'mode':'captured','host':'linux','gate':stage,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),'source_tree':inputs['tree'],'input_fingerprints':inputs['files'],'command':cmd,'cwd':str(W),'environment':settings,'started_at':started}
  (d/'captured-inputs.json').write_text(json.dumps(capture,indent=2)+'\n')
  with (d/'stdout.log').open('wb') as out,(d/'stderr.log').open('wb') as err:
   child=subprocess.Popen([PY,str(A/'dag041-child-terminal.py'),str(d)],cwd=W,env=env,stdout=out,stderr=err,start_new_session=True);peak=before.copy();stop=None
   print(json.dumps({'stage':stage,'pid':child.pid,'command':cmd}),flush=True)
   while child.poll() is None:
    time.sleep(5);u=usage()
    for k in ('target_bytes','other_bytes'):peak[k]=max(peak[k],u[k])
    peak['free_bytes']=min(peak['free_bytes'],u['free_bytes'])
    if u['target_bytes']>20*1024**3 or u['other_bytes']>4*1024**3 or u['free_bytes']<16*1024**3:
     stop='budget';os.killpg(child.pid,signal.SIGTERM);break
   rc=child.wait()
  terminal=json.loads((d/'command-exit.json').read_text()) if (d/'command-exit.json').exists() else None
  r={**capture,'finished_at':now(),'exit_code':rc,'seconds':time.monotonic()-begin,'direct_terminal':terminal,'stop_reason':stop,'source_unchanged_after':verify()==inputs,'observed_peak':peak,'logs':{n:{'path':str(d/n),'sha256':sha(d/n),'bytes':(d/n).stat().st_size} for n in ('stdout.log','stderr.log')}}
  (d/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');(A/f'dag041-gate-{stage}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'stage':stage,'exit':rc,'seconds':r['seconds'],'peak':peak}),flush=True)
  if rc or stop:raise SystemExit(rc or 1)
if __name__=='__main__':main()
