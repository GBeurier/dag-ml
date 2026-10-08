#!/usr/bin/env python3
from pathlib import Path
import datetime,hashlib,json,os,shutil,signal,subprocess,time
W=Path('/home/delete/nirs4all/_worktrees/2026-10-07-dag-metric-validation');S=Path('/mnt/c/Temp/n4a-dag041-20261008');A=Path('/home/delete/nirs4all/_audits/2026-10-07-strict-speed-diagnostic');D=Path('/dev/shm/n4a-dag041-browser-runtime-20261008');assert not D.exists();D.mkdir();(D/'owner.json').write_text(json.dumps({'owner':'/root/strict_speed_remaining_diagnostic'}))
assert shutil.disk_usage(D).free>1024**3
stage='wasm-browser-worker-hpo-resume-linux-tmp';G=S/'gates'/stage;G.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def capture():
 f=json.loads((A/'dag041-ci-source-freeze-v2.json').read_text());m={n:sha(W/n) for n in f['all_source_files']};assert all(m[n]==r['sha256'] for n,r in f['all_source_files'].items());return m
before=capture();node='/home/delete/.nvm/versions/node/v22.21.1/bin/node';chrome='/home/delete/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome'
cmd=['/home/delete/.local/bin/rtk','proxy',node,'scripts/smoke_wasm_browser_hpo.mjs',str(S/'wasm-web041')];env=os.environ.copy();settings={'TMPDIR':str(D),'CHROME_BIN':chrome};env.update(settings);env['PATH']=str(Path(node).parent)+':'+env['PATH'];(G/'command.json').write_text(json.dumps(cmd)+'\n');start=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic();reason=None;peak=0;minfree=shutil.disk_usage(D).free
with (G/'stdout.log').open('wb') as out,(G/'stderr.log').open('wb') as err:
 child=subprocess.Popen(['/home/delete/nirs4all/nirs4all/.venv/bin/python',str(A/'dag041-child-terminal.py'),str(G)],cwd=W,env=env,stdout=out,stderr=err,start_new_session=True);print(json.dumps({'stage':stage,'pid':child.pid,'command':cmd}),flush=True)
 while child.poll() is None:
  size=sum(p.stat().st_size for p in D.rglob('*') if p.is_file() and not p.is_symlink());free=shutil.disk_usage(D).free;peak=max(peak,size);minfree=min(minfree,free)
  if size>512*1024**2 or free<512*1024**2 or time.monotonic()-t>120:
   reason='bounded-ram-or-supervisor-timeout';os.killpg(child.pid,signal.SIGTERM);break
  time.sleep(.25)
 rc=child.wait()
terminal=json.loads((G/'command-exit.json').read_text()) if (G/'command-exit.json').exists() else None
r={'gate':stage,'command':cmd,'cwd':str(W),'exit_code':rc,'direct_terminal':terminal,'stop_reason':reason,'source_tree':'7e77957b89fae7097886d310e71e249dc5eb9c45','source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),'source_unchanged_after':capture()==before,'input_fingerprints':before,'environment':settings,'started_at':start,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seconds':time.monotonic()-t,'ram_peak_bytes':peak,'ram_minfree_bytes':minfree,'original_45s_deadline_unchanged':True,'wasm_bg_sha256':sha(S/'wasm-web041/dag_ml_wasm_bg.wasm'),'browser_sha256':sha(Path(chrome)),'logs':{n:{'path':str(G/n),'sha256':sha(G/n),'bytes':(G/n).stat().st_size} for n in ['stdout.log','stderr.log']}}
(G/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');(A/('dag041-gate-'+stage+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'stage':stage,'exit':rc,'peak':peak,'seconds':r['seconds']}));raise SystemExit(rc or (1 if reason else 0))
