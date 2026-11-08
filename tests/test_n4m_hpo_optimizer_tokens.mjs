// Synthetic optimizer protocol controls through the real DAG/WASM ingress.
import assert from 'node:assert/strict';
import { statSync, readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { N4mWasmHostOptimizer } from '../bindings/js/n4m_hpo_optimizer.mjs';
assert.ok(process.env.DAG_ML_WASM_PKG && path.isAbsolute(process.env.DAG_ML_WASM_PKG));
assert.ok(statSync(process.env.DAG_ML_WASM_PKG).isFile());
const require = createRequire(import.meta.url), dag = require(process.env.DAG_ML_WASM_PKG);
const { ridgeNodeResult } = require('../scripts/hpo_ridge_operator.cjs');
const graph={id:'graph:token-controls',interface:{inputs:[],outputs:[]},nodes:[{id:'model:m',kind:'model',operator:{type:'JSModel'},params:{},ports:{inputs:[],outputs:[{name:'oof',kind:'prediction',representation:null,cardinality:'one',description:''}]},metadata:{},seed_label:null}],edges:[],search_space_fingerprint:null,metadata:{}};
const manifest=JSON.parse(dag.derive_controller_manifest_json(JSON.stringify({controller_id:'controller:js-hpo-model',controller_version:'1.0.0',operator_kind:'model',input_ports:[],output_ports:graph.nodes[0].ports.outputs})));
const folds={id:'folds:tokens',sample_ids:['s1','s2'],sample_groups:{},folds:[{fold_id:'fold:0',train_sample_ids:['s2'],validation_sample_ids:['s1'],metadata:{}},{fold_id:'fold:1',train_sample_ids:['s1'],validation_sample_ids:['s2'],metadata:{}}]};
const campaign={id:'campaign:tokens',root_seed:7,split_invocation:{id:'split:outer',controller_id:null,leakage_policy:{split_unit:'sample',forbid_origin_cross_fold:true,allow_observation_split_with_shared_target:false,require_group_ids:false,unsafe_flags:[]},params:{},fold_set:folds}};
const plan=dag.build_execution_plan_json('plan:tokens',JSON.stringify(graph),JSON.stringify(campaign),JSON.stringify([manifest]));
const envelope=readFileSync(new URL('../crates/dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json',import.meta.url),'utf8');
let parameters, saved, creates=0, loads=0;
class SyntheticOptimizer {
  static create(){ creates++; return new this([]); }
  static load(){ loads++; return new this(structuredClone(saved)); }
  constructor(records){ this.records=records; }
  ask(){ const p=structuredClone(parameters); const raw={}; for(const [key,value] of Object.entries(p)){ if(Array.isArray(value)) value.forEach((item,i)=>raw[`${key}#${i}`]=item); else raw[key]=value; } this.records.push({id:BigInt(this.records.length),status:'running',parameters:raw,score:null}); return {id:BigInt(this.records.length-1),parameters:p}; }
  save(){ saved=structuredClone(this.records); return Uint8Array.of(1); }
  trialRecords(){ return this.records; }
  tell(id,status,score){ Object.assign(this.records[Number(id)],{status,score:score??null}); }
  intermediate(){ return false; }
  dispose(){}
}
const cases=[
 ['float-integral',{kind:'float',low:1,high:3},3,'3.0'],['float-window',{kind:'float',low:2e19,high:9e20},2e19,'20000000000000000000.0'],
 ['float-negzero',{kind:'float',low:-1,high:1},-0,'-0.0'],['float-zero',{kind:'float',low:-1,high:1},0,'0.0'],
 ['float-subnormal',{kind:'float',low:0,high:1},Number.MIN_VALUE,'5e-324'],['float-max',{kind:'float',low:0,high:Number.MAX_VALUE},Number.MAX_VALUE,'1.7976931348623157e+308'],
 ['log-float',{kind:'log_float',low:1,high:10},2,'2.0'],['int',{kind:'int',low:1,high:3},2n,'2'],['log-int',{kind:'log_int',low:1,high:3},3n,'3'],
 ['int-max',{kind:'int',low:1,high:9007199254740992n},9007199254740992n,'9007199254740992'],['int-min',{kind:'int',low:-9007199254740992n,high:-1},-9007199254740992n,'-9007199254740992'],
 ['cat-float',{kind:'categorical',type:'float',choices:[0,2e19]},2e19,'20000000000000000000.0'],['cat-int',{kind:'categorical',type:'integer',choices:[1n,2n]},2n,'2'],
 ['cat-bool',{kind:'categorical',type:'boolean',choices:[false,true]},true,'true'],['cat-string',{kind:'categorical',type:'string',choices:['😀']},'😀','"😀"'],
 ['ordinal-negzero',{kind:'ordinal',choices:[-0,1]},-0,'-0.0'],['ordinal-int',{kind:'ordinal',choices:[2,3]},2,'2'],['ordinal-float',{kind:'ordinal',choices:[1,2e19]},2e19,'20000000000000000000.0'],
 ['tuple-float',{kind:'sorted_tuple',length:2,low:-1,high:3},[-0,2],'[-0.0,2.0]'],['tuple-int',{kind:'sorted_tuple',length:2,low:1,high:3,integer:true},[1n,2n],'[1,2]'],
];
const config=space=>({Optimizer:SyntheticOptimizer,dagMl:dag,space,options:{},objective:{},persist:()=>{}});
let passed=0;
for(const [label,axis,value,token] of cases){
 parameters={offset:1n,probe:value}; const space={offset:{kind:'int',low:1,high:3},probe:axis};let snapshot;const adapter=new N4mWasmHostOptimizer({...config(space),persist:s=>{snapshot=structuredClone(s)}});let askRaw,taskRaw;
 const request={target_node:'model:m',trial_budget:1,metric:'rmse',direction:'minimize',optimizer_descriptor:{owner:'tokens'}};
 const result=JSON.parse(dag.host_hpo_search_json(plan,JSON.stringify([manifest]),envelope,JSON.stringify(request),undefined,(owner,task)=>{taskRaw=task;return ridgeNodeResult(owner,task,folds,'tokens')},(op,payload)=>{const reply=adapter.callback(op,payload);if(op==='ask') askRaw=reply;return reply;}));
 assert.equal(result.status,'completed',label);assert.ok(askRaw.includes(`"probe":${token}`),`${label}: ${askRaw}`);
 const expected = Array.isArray(value) ? value.map(x => typeof x === 'bigint' ? Number(x) : x) : typeof value === 'bigint' ? Number(value) : value;
 assert.deepEqual(JSON.parse(taskRaw).node_plan.params.probe,expected,label);assert.equal(snapshot.schema,'dagml.n4m.wasm-hpo.v2');assert.equal(typeof snapshot.committed_json,'string');
 const fp=JSON.parse(snapshot.committed_json).fingerprint;adapter.close();const restored=new N4mWasmHostOptimizer({...config(space),state:snapshot});assert.equal(JSON.parse(restored.committedJson).fingerprint,fp);restored.close();passed++;
 if(label==='float-negzero'){
  const tampered=structuredClone(snapshot);tampered.committed_json=tampered.committed_json.replaceAll('"probe":-0.0','"probe":0.0');
  assert.notEqual(tampered.committed_json,snapshot.committed_json);assert.throws(()=>new N4mWasmHostOptimizer({...config(space),state:tampered}),/histories disagree/);passed++;
 }
}
for(const [label,space] of [
 ['nfc',{'é':{kind:'int',low:1,high:2},'e\u0301':{kind:'int',low:1,high:2}}],
 ['name',{'\ud800':{kind:'int',low:1,high:2}}],
 ['choice',{x:{kind:'categorical',type:'string',choices:['\ud800']}}],
]){
 for(const state of [null,{schema:'dagml.n4m.wasm-hpo.v2',contract:{},n4mopt:[1],committed_json:null,prepared_json:null}]){const before=[creates,loads];assert.throws(()=>new N4mWasmHostOptimizer({...config(space),state}),undefined,label);assert.deepEqual([creates,loads],before,label);passed++;}
}
const before=[creates,loads];assert.throws(()=>new N4mWasmHostOptimizer({...config({x:{kind:'int',low:1,high:2}}),state:{schema:'dagml.n4m.wasm-hpo.v1',contract:{},n4mopt:[1]}}));assert.deepEqual([creates,loads],before);passed++;
const valid = new N4mWasmHostOptimizer(config({'e\u0301':{kind:'int',low:1,high:2}}));valid.close();passed++;
// Run the browser worker's message handler in a Node projection with its
// imports supplied locally; assert the native evaluator receives raw task text.
let forwarded;
globalThis.__workerProbe = {
  init: async () => {},
  dag: { host_hpo_evaluate_worker_fold_json: raw => { forwarded=raw; return '{}'; },
    host_hpo_evaluate_worker_task_json: raw => { forwarded=raw; return '{}'; } },
  ridge: () => '{}',
};
globalThis.self={postMessage: message => assert.ok(message.result)};
let browserSource=readFileSync(new URL('../scripts/browser_hpo/worker.js',import.meta.url),'utf8');
browserSource=browserSource.replace('import init, * as dagMl from "/pkg/dag_ml_wasm.js";', 'const {init,dag:dagMl}=globalThis.__workerProbe;')
  .replace('import {ridgeNodeResult} from "/ridge_operator.js";', 'const ridgeNodeResult=globalThis.__workerProbe.ridge;')
  .replace('"/n4m_multimodal_controller.mjs"',JSON.stringify(new URL('../bindings/js/n4m_multimodal_controller.mjs',import.meta.url).href));
await import(`data:text/javascript;base64,${Buffer.from(browserSource).toString('base64')}`);
const rawTask='{"params":{"x":-0.0,"y":20000000000000000000.0}}';
for(const packet of [rawTask,`{"kind":"complete","task":${rawTask}}`,`{"kind":"fold","fold_index":0,"task":${rawTask}}`]){
 forwarded=null;await self.onmessage({data:{taskJson:packet,manifests:'[]',envelope:'{}',request:'{}',folds:{}}});assert.equal(forwarded,rawTask);passed++;
}
delete globalThis.self;delete globalThis.__workerProbe;
assert.equal(passed,32);console.log(JSON.stringify({cases:32,passed,failed:0,skipped:0}));
