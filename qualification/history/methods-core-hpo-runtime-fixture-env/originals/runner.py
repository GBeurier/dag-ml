#!/usr/bin/env python3
from pathlib import Path
import importlib.util,os
A=Path('/home/delete/nirs4all/_audits/2026-10-07-strict-speed-diagnostic')
p=importlib.util.spec_from_file_location('gate_runner',A/'run-dag041-local-r-gates.py');m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
lib='/home/delete/nirs4all/_audits/2026-10-06-debt-extension/workflow/public-methods-134-full/n4m/lib/libn4m.so.2.17.0'
for n in ['N4M_LIBRARY_PATH','N4M_LIB_PATH','PLS4ALL_LIB_PATH']:os.environ[n]=lib
os.environ['PLS4ALL_PYTHONPATH']=str(m.S/'methods-producer-dcc570/bindings/python/src')
os.environ['N4M_ESTIMATOR_ROLES_FIXTURE']='/mnt/c/Temp/n4a-dag041-20261008/methods-producer-dcc570/parity/fixtures/estimator_roles_n4me.json'
os.environ['DAG_ML_REQUIRE_METHODS_CLI_TEST']='1';os.environ['DAG_ML_REQUIRE_N4M_TEST']='1'
m.STAGES={'methods-cli-real-replay':['cargo','test','--offline','--locked','-p','dag-ml-cli','--test','initial_full_refit','cli_replays_portable_methods_pls_payload_in_fresh_process'],'methods-core-hpo-runtime-fixture-env':['bash','scripts/test_methods_optimizer_local.sh']}
if __name__=='__main__':m.main()
