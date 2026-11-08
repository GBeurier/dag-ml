#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5] / 'examples/adapters'))
from python_process_controller import emit_description
S={'sample:1':(1.0,1.0),'sample:2':(2.0,2.0)}
FOLD={'fold:0':(['sample:2'],['sample:1']),'fold:1':(['sample:1'],['sample:2'])}
def main():
    if sys.argv[1:]==['--describe']:
        emit_description(adapter_id='probe-ridge'); return
    t=json.loads(sys.stdin.readline())
    a=t['node_plan']['params']['offset']-1
    tr,va=FOLD[t['fold_id']]
    xy=sum(S[i][0]*S[i][1] for i in tr); xx=sum(S[i][0]**2 for i in tr); w=xy/(xx+a)
    r={'node_id':t['node_plan']['node_id'],'outputs':{},'predictions':[{'producer_node':t['node_plan']['node_id'],'partition':'validation','fold_id':t['fold_id'],'sample_ids':va,'values':[[w*S[i][0]] for i in va],'target_names':['y']}],
    'regression_targets':[{'level':'sample','unit_ids':[{'level':'sample','id':i} for i in va],'values':[[S[i][1]] for i in va],'target_names':['y']}],
    'lineage':{'record_id':f"lineage:p:{t['variant_id']}:{t['fold_id']}",'run_id':t['run_id'],'node_id':t['node_plan']['node_id'],'phase':t['phase'],'controller_id':t['node_plan'].get('controller_id','controller:js-hpo-model'),'controller_version':t['node_plan']['controller_version'],'variant_id':t['variant_id'],'fold_id':t['fold_id'],'branch_path':t['branch_path'],'input_lineage':[],'artifact_refs':[],'params_fingerprint':t['node_plan']['params_fingerprint'],'data_model_shape_fingerprint':None,'aggregation_policy_fingerprint':None,'seed':t.get('seed'),'unsafe_flags':[],'metrics':{},'loss_attestations':[],'early_stopping_records':[]}}
    print(json.dumps(r))
main()
