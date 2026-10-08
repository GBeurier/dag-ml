#!/usr/bin/env python3
from pathlib import Path
import importlib.util,os
A=Path('/home/delete/nirs4all/_audits/2026-10-07-strict-speed-diagnostic')
p=importlib.util.spec_from_file_location('gate_runner',A/'run-dag041-local-r-gates.py');m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
S=m.S; W=m.W; D=S/'octave-bindings'; B=S/'octave-elf-cache/bin'
os.environ['PATH']=str(B)+':'+os.environ['PATH']
os.environ['OCTAVE_PATH']=str(D)
os.environ['DAGML_REQUIRE_HPO_MATLAB']='1';os.environ['DAGML_REQUIRE_HPO_OCTAVE']='1'
m.STAGES={
'octave-mex-loss':[str(B/'mkoctfile'),'--mex',str(W/'bindings/matlab/native/task_training_loss_binding.c'),'-o',str(D/'+dagml/taskTrainingLossBindingNative.mex')],
'octave-mex-phase':[str(B/'mkoctfile'),'--mex',str(W/'bindings/matlab/native/execution_plan_phase.c'),'-o',str(D/'+dagml/executeExecutionPlanPhaseNative.mex')],
'octave-native-registry':[str(B/'octave'),'--no-gui','--quiet','--eval',f"addpath('{D}/tests'); local_implementation_registry"],
'octave-host-hpo-wrapper':[str(B/'octave'),'--no-gui','--quiet','--eval',f"addpath('{D}/tests'); host_hpo_search"],
'r-octave-hpo-adapters':[m.PY,'scripts/test_hpo_language_adapters.py'],
'octave-native-hpo-resume':['cargo','test','--offline','--locked','-p','dag-ml-cli','host_hpo_cli_runs_parallel_pruning_and_resumes_native_checkpoint'],
'octave-ridge-folds-refit-replay':['cargo','test','--offline','--locked','-p','dag-ml-cli','--test','octave_hpo_ridge'],
}
for name in ['initial_full_refit','cv_refit_predict','replay_bundle']:
 m.STAGES['octave-'+name.replace('_','-')]=[str(B/'octave'),'--no-gui','--quiet','--eval',f"addpath('{D}/tests'); {name}"]
if __name__=='__main__':m.main()
