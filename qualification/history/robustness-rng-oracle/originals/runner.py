#!/usr/bin/env python3
from pathlib import Path
import importlib.util,os
A=Path('/home/delete/nirs4all/_audits/2026-10-07-strict-speed-diagnostic')
p=importlib.util.spec_from_file_location('gate_runner',A/'run-dag041-local-r-gates.py');m=importlib.util.module_from_spec(p);p.loader.exec_module(m)
os.environ['NIRS4ALL_LEGACY_REPO']='/home/delete/nirs4all/nirs4all'
m.STAGES={
'archive-v1-legacy-security':[m.PY,'-m','unittest','discover','-s','tests','-p','test_archive_v1_contract.py'],
'conformal-robustness-oracle':[m.PY,'-m','pytest','--capture=sys','parity/conformal/tests/test_conformal_robustness_contracts.py','-q'],
'training-contract-oracle':[m.PY,'-m','pytest','--capture=sys','parity/training/tests/test_training_contracts.py','-q'],
'robustness-rng-oracle':[m.PY,'-m','pytest','--capture=sys','parity/robustness_rng/tests/test_robustness_rng_contract.py','-q'],
'canonical-rust-oracle':['cargo','test','--offline','--locked','--manifest-path','parity/canonical/rust-oracle/Cargo.toml'],
'canonical-cross-language-oracle':[m.PY,'-m','pytest','--capture=sys','parity/canonical/tests/test_rust_oracle_parity.py','-q'],
}
if __name__=='__main__':m.main()
