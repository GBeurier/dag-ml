#!/usr/bin/env python3
"""Authorized bounded local R/native gates, exact frozen041 sources, first results retained."""
from pathlib import Path
import argparse,ctypes,hashlib,json,os,shutil,signal,subprocess,time
W=Path('/home/delete/nirs4all/_worktrees/2026-10-07-dag-metric-validation')
S=Path('/mnt/c/Temp/n4a-dag041-20261008')
A=Path('/home/delete/nirs4all/_audits/2026-10-07-strict-speed-diagnostic')
PY='/home/delete/nirs4all/nirs4all/.venv/bin/python'
R='/home/delete/miniconda3/bin/R'
RS='/home/delete/miniconda3/bin/Rscript'
LIB=S/'target/release/libdag_ml_capi.so'
STAGES={
 'r-capi-release':['cargo','build','--offline','--locked','-p','dag-ml-capi','--release'],
 'r-source-package-build':[R,'CMD','build',str(W/'bindings/r')],
 'r-source-package-install':[R,'CMD','INSTALL','--library='+str(S/'r-source-local/library'),str(S/'r-source-local/dagml_0.3.41.tar.gz')],
 'r-source-package-check':[R,'CMD','check','--no-manual',str(S/'r-source-local/dagml_0.3.41.tar.gz')],
 'r-seed-json':[RS,'--vanilla',str(W/'scripts/test_r_seed_json.R')],
 'r-prospectr-adapter':['cargo','test','--offline','--locked','-p','dag-ml-cli','--test','cli_contracts','prospectr_process_controller'],
 'r-mdatools-adapter':['cargo','test','--offline','--locked','-p','dag-ml-cli','--test','cli_contracts','mdatools_process_controller'],
 'r-hpo-adapter':[PY,'scripts/test_hpo_language_adapters.py'],
 'r-native-hpo-resume':['cargo','test','--offline','--locked','-p','dag-ml-cli','host_hpo_cli_runs_parallel_pruning_and_resumes_native_checkpoint'],
 'r-native-ridge-folds-refit-replay':['cargo','test','--offline','--locked','-p','dag-ml-cli','--test','r_hpo_ridge'],
}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_sources():
 f=json.loads((A/'dag041-ci-source-freeze-v1.json').read_text())
 for n,row in f['all_tracked_plus_new_source_files'].items():assert sha(W/n)==row['sha256'],(n,'sourcefreezechanged')
 return f['virtual_full_git_tree']
def usage():
 sizes={'target_bytes':0,'other_bytes':0,'free_bytes':shutil.disk_usage(S).free}
 for directory,_,files in os.walk(S):
  for n in files:
   p=Path(directory)/n
   try:size=p.stat().st_size
   except FileNotFoundError:continue
   k='target_bytes' if p.is_relative_to(S/'target') else 'other_bytes';sizes[k]+=size
 return sizes
def main():
 parser=argparse.ArgumentParser();parser.add_argument('stages',nargs='+',choices=STAGES);args=parser.parse_args()
 assert json.loads((S/'owner.json').read_text())['owner']=='/root/strict_speed_remaining_diagnostic'
 (S/'r-source-local/library').mkdir(parents=True,exist_ok=True)
 e=os.environ.copy();e.pop('RUSTFLAGS',None)
 e.update({'CARGO_TARGET_DIR':str(S/'target'),'TMPDIR':str(S/'tmp'),'CARGO_INCREMENTAL':'0','CARGO_PROFILE_DEV_DEBUG':'0','CARGO_PROFILE_TEST_DEBUG':'0','CARGO_NET_OFFLINE':'true','PYO3_PYTHON':PY,'DAGML_NATIVE_LIBRARY':str(LIB),'DAGML_REQUIRE_HPO_R':'1','DAG_ML_REQUIRE_R_ADAPTER_TEST':'1','R_LIBS':str(S/'r-source-local/library'),'DAG_ML_DATA_REPO':'/home/delete/nirs4all/dag-ml-data'})
 e['PATH']='/home/delete/miniconda3/bin:'+str(Path(PY).parent)+':'+e.get('PATH','')
 for stage in args.stages:
  tree=verify_sources();before=usage();assert before['free_bytes']>=16*1024**3
  d=S/'gates'/stage;d.mkdir()
  cwd=S/'r-source-local' if stage in ('r-source-package-build','r-source-package-install','r-source-package-check') else W
  cmd=['/home/delete/.local/bin/rtk','proxy',*STAGES[stage]]
  (d/'command.json').write_text(json.dumps(cmd)+'\n')
  start=time.monotonic()
  with (d/'stdout.log').open('wb') as out,(d/'stderr.log').open('wb') as err:
   child=subprocess.Popen([PY,str(A/'dag041-child-terminal.py'),str(d)],cwd=cwd,env=e,stdout=out,stderr=err,start_new_session=True)
   print(json.dumps({'stage':stage,'pid':child.pid,'command':cmd,'cwd':str(cwd)}),flush=True)
   peak=before.copy();reason=None
   while child.poll() is None:
    time.sleep(5);u=usage()
    peak['target_bytes']=max(peak['target_bytes'],u['target_bytes']);peak['other_bytes']=max(peak['other_bytes'],u['other_bytes']);peak['free_bytes']=min(peak['free_bytes'],u['free_bytes'])
    if u['target_bytes']>20*1024**3 or u['other_bytes']>4*1024**3 or u['free_bytes']<16*1024**3:
     reason='owned-storage-budget';os.killpg(child.pid,signal.SIGTERM);break
   code=child.wait()
  terminal=json.loads((d/'command-exit.json').read_text()) if (d/'command-exit.json').exists() else None
  result={'gate':stage,'command':cmd,'cwd':str(cwd),'exit_code':code,'direct_terminal':terminal,'stop_reason':reason,'seconds':time.monotonic()-start,'observed_peak':peak,'source_tree':tree,'source_unchanged_after':verify_sources()==tree,'logs':{n:{'path':str(d/n),'sha256':sha(d/n),'bytes':(d/n).stat().st_size} for n in ('stdout.log','stderr.log')}}
  (d/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');(A/f'dag041-gate-{stage}.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps({'stage':stage,'exit_code':code,'peak':peak}),flush=True)
  if code or reason:raise SystemExit(code or 1)
if __name__=='__main__':main()
