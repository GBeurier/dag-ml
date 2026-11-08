// Actual WASM public-wrapper controls. Synthetic host only; no numerical/IO claim.
import assert from 'node:assert/strict';
import { readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
assert.ok(process.env.DAG_ML_WASM_PKG && path.isAbsolute(process.env.DAG_ML_WASM_PKG), 'DAG_ML_WASM_PKG must be an absolute file path');
assert.ok(statSync(process.env.DAG_ML_WASM_PKG).isFile(), 'DAG_ML_WASM_PKG must be a file');
const require=createRequire(import.meta.url), dag=require(process.env.DAG_ML_WASM_PKG);
const root=new URL('../crates/dag-ml-core/tests/fixtures/multi_output_target_claims/',import.meta.url);
const read=n=>JSON.parse(readFileSync(new URL(n,root),'utf8'));
const raw=readFileSync(new URL('explain/package.json',root),'utf8'),pkg=JSON.parse(raw),trusted=read('explain/training-request.json').controller_manifests;
const cases=[
 ['S00','PREDICT','normal',true],['S01-v1','PREDICT','v1',true],['S02-old','PREDICT','old',false],['S03-alien','PREDICT','alien',false],
 ['S04-duplicate','PREDICT','duplicate',false],['S05-partial','PREDICT','partial',true],['S07-extra','PREDICT','superset',false],['S08-reordered','PREDICT','reordered',true],
 ['V1-forged-preimage','PREDICT','forged',false],['V2-development-precedence','PREDICT','development',true],
 ['C08-selected','EXPLAIN','normal',true],['C02-missing-port','EXPLAIN','missing-port',true],['C05-global','EXPLAIN','global',true],
 ['C01-wrong-target','EXPLAIN','target',false],['C03-wrong-producer','EXPLAIN','producer',false],['C10-empty-selection','EXPLAIN','empty',false],['class-alias','EXPLAIN','alias',false],
 ['P00','EXPLAIN','normal',true],['P01-handle','EXPLAIN','handle',false],['P02-nested-handle','EXPLAIN','nested-handle',false],
 ['P03-raw-features','EXPLAIN','raw_features',false],['P04-feature-matrix','EXPLAIN','feature_matrix',false],['P05-raw-spectra','EXPLAIN','raw_spectra',false],['P06-raw-wavelengths','EXPLAIN','raw_wavelengths',false],
 ['P07-unselected-raw','EXPLAIN','unselected-raw',false],['P10-arbitrary-key-matrix','EXPLAIN','matrix',true],
 ['P08b-raw-over-u64','EXPLAIN','token:18446744073709551616',false],['integer-u64-max','EXPLAIN','token:18446744073709551615',true],['integer-i64-min','EXPLAIN','token:-9223372036854775808',true],
 ['exponent-large','EXPLAIN','token:1e300',true],['exponent-small','EXPLAIN','token:-2.5e-300',true],['exponent-max','EXPLAIN','token:1.7976931348623157e308',true],
 ['JSON.stringify(2e19)','EXPLAIN','jsfloat',false],
];
// Insert after JSON.stringify so raw surrogates reach the callback boundary.
const utf16Tokens = {
 'raw-high':'"'+'\ud800'+'"', 'raw-low':'"'+'\udc00'+'"',
 'escaped-high':'"\\ud800"', 'escaped-low':'"\\udc00"',
 emoji:'"😀"', normal:'"normal"', replacement:'"�"',
};
for(const name of Object.keys(utf16Tokens)){
 cases.push(['utf16-node-'+name,'EXPLAIN','utf16-node:'+name,!name.includes('high')&&!name.includes('low')]);
 // Portable result fields are closed; legal strings reach the unknown-field refusal.
 cases.push(['utf16-portable-'+name,'EXPLAIN','utf16-portable:'+name,false]);
}
cases.push(['utf16-node-throw','EXPLAIN','utf16-node-throw',false]);
cases.push(['utf16-portable-throw','EXPLAIN','utf16-portable-throw',false]);

const portableFailureNames=['high','low','replacement','normal','number','object','error-high','throwing-name','throwing-stack','hostile-tag','proxy','revoked-proxy','cyclic-array','self-doubled-array'];
for(const name of portableFailureNames)cases.push(['portable-failure-'+name,'EXPLAIN','portable-failure:'+name,false]);
function portableFailure(name){
 if(name==='high')return '\ud800';if(name==='low')return '\udc00';if(name==='replacement')return '�';if(name==='normal')return 'plain';if(name==='number')return 42;if(name==='object')return {value:1};if(name==='error-high')return new Error('\ud800');
 if(name==='throwing-name'||name==='throwing-stack'){const error=new Error('controlled');Object.defineProperty(error,name==='throwing-name'?'name':'stack',{get(){throw new Error('hostile getter')}});return error;}
 if(name==='hostile-tag'){const object={};Object.defineProperty(object,Symbol.toStringTag,{get(){throw new Error('hostile tag')}});return object;}
 if(name==='proxy')return new Proxy({},{get(){throw new Error('hostile proxy')}});
 if(name==='revoked-proxy'){const pair=Proxy.revocable({},{});pair.revoke();return pair.proxy;}
 const array=[];array.push(array);if(name==='self-doubled-array')array.push(array);return array;
}

assert.equal(cases.length,63,'contextual replay case count');
for(const [label,phase,mode,expected] of cases){
 const envelopes=read('shared/envelopes.json');let invoke=0,hydrate=0,release=0,next=1;
 for(const [key,e] of Object.entries(envelopes)){
  const old=e.predict_cohort;
  if(mode==='v1'||mode==='forged'){e.schema_version=1;delete e.predict_cohort;if(mode==='forged')e.coordinator_relations.records[0].sample_id='sample:alien';}
  else{
   const names=phase==='EXPLAIN'?['protein']:['protein','fat'];
   envelopes[key]=JSON.parse(dag.attach_predict_cohort_to_envelope_json(JSON.stringify(e),JSON.stringify({role:'inference',relations:old.relations,target_names:names,data_content_fingerprint:old.data_content_fingerprint})));
   if(mode==='development'){
    const dev=structuredClone(e.coordinator_relations);dev.records.forEach((r,i)=>{r.sample_id='sample:'+(i+1);r.observation_id='observation:dev:'+(i+1);});
    envelopes[key].coordinator_relations=dev;envelopes[key].relation_fingerprint=dag.sample_relation_set_fingerprint_json(JSON.stringify(dev));
   }
  }
 }
 const request=JSON.parse(dag.sign_training_replay_request_json(JSON.stringify({...read('shared/request.json'),phase,source_outcome_fingerprint:pkg.training_outcome.outcome_fingerprint,output_binding_ids:phase==='EXPLAIN'?['output:a']:['output:a','output:b']})));
 const callback=(owner,json)=>{
  const t=JSON.parse(json);
  if(t.operation==='hydrate_artifact_payload'){
   hydrate++;
   if(mode==='utf16-portable-throw')throw new Error('controlled portable failure');
   if(mode.startsWith('portable-failure:'))throw portableFailure(mode.slice(17));
   const text=JSON.stringify({operation:'hydrated_artifact_payload',schema_version:1,handle:{handle:next++,kind:'artifact',owner_controller:owner}});
   return mode.startsWith('utf16-portable:') ? text.replace(/^\{/,`{"probe":${utf16Tokens[mode.slice(15)]},`) : text;
  }
  if(t.operation==='release_hydrated_artifact_payload'){release++;return JSON.stringify({operation:'released_hydrated_artifact_payload',schema_version:1});}
  if(mode==='utf16-node-throw')throw new Error('controlled node failure');
  invoke++;const n=t.node_plan,model=n.kind==='model',port=model?'oof':'x_out',targets=model?pkg.output_bindings.find(b=>b.node_id===n.node_id).target_names:[];
  let ids=['sample:11','sample:12','sample:13','sample:14'];
  if(mode==='old')ids[0]='sample:1';if(mode==='alien')ids[0]='sample:alien';if(mode==='duplicate')ids[1]=ids[0];if(mode==='partial')ids=ids.slice(0,1);if(mode==='reordered')ids.reverse();if(mode==='superset')ids.push('sample:alien');
  const result={node_id:n.node_id,outputs:{[port]:{handle:next++,kind:model?'prediction':'data',owner_controller:owner}},predictions:model&&phase==='PREDICT'?[{producer_node:n.node_id,producer_port:port,partition:'final',fold_id:null,sample_ids:ids,values:ids.map(()=>targets.map(()=>0)),target_names:targets}]:[],explanations:model&&phase==='EXPLAIN'?[{producer_node:n.node_id,producer_port:port,method:'synthetic',target_name:targets[0],payload:{importance:0}}]:[],lineage:{record_id:'lineage:contextual:'+next++,run_id:t.run_id,node_id:n.node_id,phase,controller_id:owner,controller_version:n.controller_version,variant_id:t.variant_id,fold_id:t.fold_id,branch_path:t.branch_path,input_lineage:[],artifact_refs:[],params_fingerprint:n.params_fingerprint,data_model_shape_fingerprint:null,aggregation_policy_fingerprint:null,seed:null,unsafe_flags:[],metrics:{},loss_attestations:[],early_stopping_records:[]}};
  for(const b of result.explanations){
   if(mode.startsWith('utf16-node:'))b.payload={probe:'UTF16_PROBE_TOKEN'};
   if(mode==='missing-port')delete b.producer_port;if(mode==='global')b.target_name=null;if(mode==='target')b.target_name='wrong';if(mode==='alias')b.target_name='protein:a';if(mode==='producer')b.producer_node='model:unknown';
   if(mode==='handle')b.payload={handle:1};if(mode==='nested-handle')b.payload={nested:[{model_handle:1}]};
   if(['raw_features','feature_matrix','raw_spectra','raw_wavelengths'].includes(mode))b.payload={nested:[{[mode.toUpperCase()]:[[0]]}]};
   if(mode==='unselected-raw'&&n.node_id==='model:second')b.payload={Raw_Spectra:[0]};if(mode==='matrix')b.payload={X:[[1,2],[3,4]]};
   if(mode.startsWith('token:'))b.payload={n:'RAW_NUMERIC_TOKEN'};if(mode==='jsfloat')b.payload={n:2e19};
  }
  if(mode==='empty'&&n.node_id==='model:base')result.explanations=[];
  let text=JSON.stringify(result);if(mode.startsWith('utf16-node:'))text=text.replace('"UTF16_PROBE_TOKEN"',utf16Tokens[mode.slice(11)]);if(mode.startsWith('token:'))text=text.replace('"RAW_NUMERIC_TOKEN"',mode.slice(6));return text;
 };
 let ok=false,error=null,outcome,outcomeRaw;
 try{outcomeRaw=dag.replay_training_package_json(raw,JSON.stringify(request),JSON.stringify(envelopes),JSON.stringify(trusted),'outcome:contextual','run:contextual',callback);outcome=JSON.parse(outcomeRaw);ok=true;}catch(e){error=String(e);}
 assert.equal(ok,expected,`${label}: ${error}`);
 if(ok&&phase==='EXPLAIN'){assert.equal(outcome.explanations.length,1);assert.equal(outcome.lineage.length,3);}
 if(ok&&['token:18446744073709551615','token:-9223372036854775808'].includes(mode))assert.ok(outcomeRaw.includes('\"n\":'+mode.slice(6)),label+' exact output token');
 if(mode==='forged')assert.equal(invoke,0);
 if(mode.startsWith('utf16-')&&!expected){
  if(mode.includes('raw-high')||mode.includes('raw-low'))assert.match(error,/not valid UTF-16/);
  else assert.doesNotMatch(error,/not valid UTF-16/);
 }
 if(mode.endsWith('-throw'))assert.match(error,/controlled (portable|node) failure/);
 if(mode.startsWith('portable-failure:')){
  assert.match(error,/rejected portable artifact operation:/);
  if(['portable-failure:high','portable-failure:low','portable-failure:error-high'].includes(mode))assert.match(error,/ill_formed_utf16/);
  if(mode==='portable-failure:replacement')assert.ok(error.includes('�'));
  if(mode.includes('hostile-tag')||mode.endsWith(':proxy')||mode.includes('revoked-proxy'))assert.match(error,/unrenderable JS value/);
  if(mode.includes('throwing-name'))assert.match(error,/unreadable/);
  // An Error renders as `name: message`; a hostile stack getter is never read.
  if(mode.includes('throwing-stack')){assert.ok(error.includes('JsValue(Error: controlled)'),error);assert.doesNotMatch(error,/unreadable|hostile getter/);}
  if(mode.includes('cyclic-array')||mode.includes('self-doubled-array'))assert.match(error,/\.\.\./);
 }
 if(mode.startsWith('token:')&&!expected)assert.match(error,/integer|range|TCV1|strict/i);
 if(mode.startsWith('utf16-portable')||mode.startsWith('portable-failure:'))assert.equal(hydrate,release+1);
 else assert.equal(hydrate,release);
 console.log(JSON.stringify({label,expected,ok,error,invoke,hydrate,release}));
}
console.log(JSON.stringify({cases:cases.length,passed:cases.length,failed:0,skipped:0}));
