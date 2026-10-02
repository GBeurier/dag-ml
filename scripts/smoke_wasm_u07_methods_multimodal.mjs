#!/usr/bin/env node
/** Real Methods WASM raw U07; worker mode also serves standard process frames. */
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";
import {createRequire} from "node:module";
import {pathToFileURL} from "node:url";
import {N4mWasmMultimodalController, SOURCE_ORDER, multimodalManifest, strictJson} from "../bindings/js/n4m_multimodal_controller.mjs";
import {replayMultimodalArchive} from "../bindings/js/multimodal_archive.mjs";

const [mode, inputPath, methodsPath, dagPath, outputPath] = process.argv.slice(2);
assert.ok(["--worker", "--campaign"].includes(mode), "--worker CONFIG METHODS_DIST or --campaign CAPTURE METHODS_DIST DAG_WASM OUTPUT required");
const methods = await import(pathToFileURL(path.resolve(methodsPath,"index.js")).href);
await methods.loadModule();
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const wireText=fs.readFileSync(inputPath,"utf8"),wire=strictJson(wireText);
const config = mode === "--worker" ? wire : {operators:wire.operators,node_params:wire.node_params,source_ids:wire.source_ids,sources:wire.sources,targets:wire.targets,target_names:wire.target_names,allow_fit:true};
function makeController(input, allowFit) {
  for(const name of SOURCE_ORDER){const source=input.sources[name];assert.ok(Array.isArray(source.sample_ids)&&source.sample_ids.length>0&&new Set(source.sample_ids).size===source.sample_ids.length&&source.sample_ids.every(id=>typeof id==="string"&&id.length>0),"Unique current source IDs required");assert.equal(source.shape[0],source.sample_ids.length);}
  const positions=Object.fromEntries(SOURCE_ORDER.map(name=>[name,new Map(input.sources[name].sample_ids.map((id,index)=>[id,index]))]));
  const targetPositions=allowFit ? new Map(input.targets.sample_ids.map((id,index)=>[id,index])) : null;
  return new N4mWasmMultimodalController({methods,operators:input.operators,nodeParams:input.node_params,sourceIds:input.source_ids,targetNames:input.target_names,digest:hash,allowFit,controllerId:input.controller_id??"controller:methods.wasm.multimodal",
    resolveFeatures:({view})=> {
      const blocks={};
      for(const name of SOURCE_ORDER) {
        const source=input.sources[name],rows=view.sample_ids.map(id=>positions[name].get(id));assert.ok(rows.every(index=>index!==undefined),"Unknown current source sample ID");
        if(name==="metadata")blocks[name]=rows.map(index=>source.rows[index]);
        else {const width=source.shape.slice(1).reduce((a,b)=>a*b,1);assert.equal(source.data.length,source.shape.reduce((a,b)=>a*b,1));assert.ok(["float32","float64"].includes(source.descriptor.dtype),"Declared numeric source dtype required");const TensorArray=source.descriptor.dtype==="float32"?Float32Array:Float64Array;const data=new TensorArray(rows.length*width);rows.forEach((index,row)=>data.set(source.data.slice(index*width,(index+1)*width),row*width));blocks[name]={data,shape:[rows.length,...source.shape.slice(1)]};}
      }
      return {sampleIds:[...view.sample_ids],blocks,sourceSchemas:Object.fromEntries(SOURCE_ORDER.map(name=>[name,input.sources[name].descriptor]))};
    },
    resolveTargets:allowFit ? ({sampleIds})=>({sampleIds,matrix:{data:Float64Array.from(sampleIds.map(id=>{const index=targetPositions.get(id);assert.notEqual(index,undefined,"Unknown target ID");return input.targets.values[index][0];})),rows:sampleIds.length,cols:1}}) : null});
}
function close(controller) {controller.close();assert.equal(controller.models.size,0);assert.equal(controller.artifacts.size,0);}
function compareNumbers(actual, expected, label) {
  assert.equal(actual.length,expected.length,label);
  actual.forEach((value,index)=>assert.ok(typeof value==="number"&&Number.isFinite(value)&&Number.isFinite(expected[index])&&Math.abs(value-expected[index])<=1e-7*Math.max(1,Math.abs(expected[index])),`${label}/${index}`));
}
function compareScores(actual, expected) {
  const coordinates=["producer_node","producer_port","variant_id","partition","fold_id","level"];
  const index=value=> {
    const result=new Map(value.reports.map(report=>[JSON.stringify(coordinates.map(key=>report[key]??null)),report]));
    assert.equal(result.size,value.reports.length,"Native score coordinates must be unique");
    return result;
  };
  const left=index(actual),right=index(expected);
  assert.deepEqual([...left.keys()].sort(),[...right.keys()].sort(),"Native score coordinates changed across hosts");
  for(const [key,report] of right) {
    const found=left.get(key);
    assert.equal(found.row_count,report.row_count,"Native score row coverage changed");
    assert.deepEqual(found.target_names,report.target_names,"Native score target coverage changed");
    assert.deepEqual(Object.keys(found.metrics).sort(),Object.keys(report.metrics).sort(),"Native metric coverage changed");
    for(const metric of Object.keys(report.metrics))compareNumbers([found.metrics[metric]],[report.metrics[metric]],`score ${key}/${metric}`);
  }
}
function replayPredictionValues(dag, outcome, sampleIds, targetNames) {
  assert.equal(outcome.outputs.length,1,"One native output binding required");
  const output=outcome.outputs[0];
  assert.equal(output.predictions.length,1,"One native prediction block required");
  assert.deepEqual(output.observation_predictions??[],[],"Unexpected observation prediction blocks");
  assert.deepEqual(output.aggregated_predictions??[],[],"Unexpected aggregated prediction blocks");
  const block=output.predictions[0];
  assert.deepEqual(block.target_names,targetNames,"Native prediction targets changed");
  assert.equal(targetNames.length,1,"Closed U07 profile requires one target");
  assert.equal(block.values.length,block.sample_ids.length,"Native prediction row coverage changed");
  for(const row of block.values)assert.ok(Array.isArray(row)&&row.length===1&&typeof row[0]==="number"&&Number.isFinite(row[0]),"Finite native Nx1 prediction required");
  for(const ids of [sampleIds,block.sample_ids])assert.ok(ids.every(id=>typeof id==="string"&&id.length>0),"Nonempty native sample IDs required");
  const producer=output.binding.node_id;
  const alignment=JSON.parse(dag.align_named_source_rows_json(JSON.stringify({sample_ids:sampleIds,required_source_ids:[producer],sources:[{source_id:producer,sample_ids:block.sample_ids}]})));
  return alignment.sources[0].row_indices.map(index=>block.values[index][0]);
}

if(mode === "--worker") {
  const ownerHost=["python","wasm","r","octave"].find(host=>(config.controller_id??"controller:methods.wasm.multimodal")===`controller:methods.${host}.multimodal`);assert.ok(ownerHost);assert.deepEqual(config.manifest,multimodalManifest(ownerHost));assert.deepEqual(config.trusted_manifest,multimodalManifest(ownerHost));
  assert.equal(config.allow_fit,config.targets!==null);
  const controller=makeController(config,config.allow_fit),originalFit=methods.MultimodalPipeline.prototype.fit;
  if(!config.allow_fit)methods.MultimodalPipeline.prototype.fit=()=>{throw new Error("Fresh archive replay cannot FIT");};
  const reader=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
  try {
    for await(const line of reader) {
      try {
        const frame=strictJson(line);assert.equal(frame.schema_version,1);
        if(frame.type==="init") {assert.equal(frame.controller_id,controller.controllerId);process.stdout.write(JSON.stringify({type:"ack",schema_version:1,status:"initialized",runtime:{execution_host:"wasm",signed_controller:controller.controllerId,node:process.version,methods_dist:path.resolve(methodsPath),methods_wasm_sha256:hash(fs.readFileSync(path.join(methodsPath,"n4m.wasm")))}})+"\n");}
        else if(frame.type==="task") {
          // Extract the original task token; do not round its u64 seed through Number.
          const task=strictJson(line,"task");
          const result=controller.invoke(controller.controllerId,task);
          process.stdout.write('{"type":"result","schema_version":1,"result":'+result+'}\n');
        } else if(frame.type==="portable_artifact") {const result=controller.invoke(controller.controllerId,JSON.stringify(frame.task));process.stdout.write('{"type":"portable_artifact","schema_version":1,"result":'+result+'}\n');}
        else {assert.equal(frame.type,"close");close(controller);process.stdout.write('{"type":"ack","schema_version":1,"status":"closed"}\n');break;}
      } catch(error) {process.stdout.write(JSON.stringify({type:"error",schema_version:1,error:{code:"wasm_methods_multimodal_refusal",message:String(error.message??error)}})+"\n");}
    }
  } finally {methods.MultimodalPipeline.prototype.fit=originalFit;close(controller);if(config.audit_path)fs.writeFileSync(config.audit_path,controller.audit.map(item=>JSON.stringify(item)).join("\n")+"\n");}
} else {
  assert.ok(typeof JSON.rawJSON==="function"&&typeof JSON.isRawJSON==="function","Lossless native numeric JSON transport is required");
  const losslessJson=text=> {strictJson(text);return JSON.parse(text,(key,value,context)=>typeof value==="number"?JSON.rawJSON(context.source):value);};
  const transport=losslessJson(wireText);
  const metadata=JSON.parse(fs.readFileSync(path.join(dagPath,"package.json"),"utf8")),require=createRequire(import.meta.url);
  const dag=metadata.type==="module"||metadata.module?await import(pathToFileURL(path.resolve(dagPath,"dag_ml_wasm.js")).href):require(path.resolve(dagPath,"dag_ml_wasm.js"));
  if(metadata.type==="module"||metadata.module)dag.initSync({module:fs.readFileSync(path.join(dagPath,"dag_ml_wasm_bg.wasm"))});
  const owner=value=>JSON.isRawJSON(value)?value:Array.isArray(value)?value.map(owner):value&&typeof value==="object"?Object.fromEntries(Object.entries(value).map(([key,child])=>[key,owner(child)])):value==="controller:methods.python.multimodal"?"controller:methods.wasm.multimodal":value;
  const manifest=multimodalManifest(),request=owner(transport.training_request);request.controller_manifests=[manifest];
  const controller=makeController(config,true);let complete;
  try {complete=JSON.parse(dag.execute_training_json(dag.sign_training_request_json(JSON.stringify(request)),JSON.stringify(transport.training_inputs.data_envelopes),JSON.stringify(transport.training_inputs.relations),"package:u07.wasm","outcome:u07.wasm","run:u07.wasm","bundle:u07.wasm",controller.callback));}finally{close(controller);}
  const outcome=JSON.parse(complete.training_outcome_json),packageValue=JSON.parse(complete.portable_predictor_package_json);
  compareScores(outcome.score_set,wire.training_outcome.score_set);
  assert.equal(packageValue.artifact_bindings.length,1);assert.equal(packageValue.execution_bundle.refit_artifacts[0].artifact.kind,"methods_multimodal_pipeline");
  const assembled=JSON.parse(dag.build_archive_v2_native_portable_payloads_json("archive:u07.wasm",complete.training_outcome_json,complete.portable_predictor_package_json));assert.equal(assembled.manifest.payloads.methods.multimodal_pipelines.length,1);
  const searchArgs=owner(transport.search_request);searchArgs.controller_manifests=[manifest];
  const compiled=losslessJson(dag.compile_pipeline_dsl_artifact_with_controllers_json(JSON.stringify(searchArgs.dsl),JSON.stringify([manifest])));
  const plan=dag.build_execution_plan_json("plan:u07.wasm.hpo",JSON.stringify(compiled.graph),JSON.stringify(compiled.campaign_template),JSON.stringify([manifest]));
  const proposals=transport.search_result.trials.map(trial=>trial.params);assert.equal(proposals.length,8);
  const searchController=makeController(config,true);let search;
  try {search=JSON.parse(dag.host_hpo_search_json(plan,JSON.stringify([manifest]),JSON.stringify(searchArgs.envelope),JSON.stringify(searchArgs.request),undefined,searchController.callback,(operation,json)=>{const message=JSON.parse(json);return JSON.stringify(operation==="ask"?{params:proposals[message.trial_index]??null}:operation==="report_intermediate"?{prune:false}:operation==="checkpoint"?{continue:true}:{ok:true});}));}finally{close(searchController);}
  const replayConfig={...config,sources:wire.prediction_sources,targets:null,allow_fit:false};const replayController=makeController(replayConfig,false);
  const replayRequest={...transport.prediction_request};replayRequest.source_outcome_fingerprint=packageValue.training_outcome.outcome_fingerprint;
  const originalFit=methods.MultimodalPipeline.prototype.fit;methods.MultimodalPipeline.prototype.fit=()=>{throw new Error("Loaded complete predictor cannot FIT");};let replay;
  try {replay=JSON.parse(dag.replay_training_package_json(complete.portable_predictor_package_json,dag.sign_training_replay_request_json(JSON.stringify(replayRequest)),JSON.stringify(transport.prediction_envelopes),JSON.stringify([manifest]),"outcome:u07.wasm.replay","run:u07.wasm.replay",replayController.callback));assert.equal(replayController.models.size,0);}finally{methods.MultimodalPipeline.prototype.fit=originalFit;close(replayController);}
  assert.deepEqual(replayController.audit.map(row=>row.operation),["hydrate","PREDICT","dispose","release"]);
  compareNumbers(replayPredictionValues(dag,replay,wire.prediction_sources.nir.sample_ids,wire.target_names),wire.expected_prediction,"Fresh raw-input replay");
  assert.ok(process.env.DAG_ML_CORE_JS_ENTRY,"Actual freshly built Core JS entry is mandatory for the archive consumer");
  const core=await import(pathToFileURL(path.resolve(process.env.DAG_ML_CORE_JS_ENTRY)).href);
  const producerController=makeController({...replayConfig,controller_id:"controller:methods.python.multimodal"},false);
  let producerArchive;
  methods.MultimodalPipeline.prototype.fit=()=>{throw new Error("Unchanged producer archive cannot FIT");};
  try {producerArchive=await replayMultimodalArchive({archiveBytes:Uint8Array.from(fs.readFileSync(wire.archive)),readPortableArchiveV2:core.readPortableArchiveV2,dagMl:dag,controller:producerController,dataEnvelopes:transport.prediction_envelopes,request:transport.prediction_request,outcomeId:"outcome:u07.python-in-wasm",runId:"run:u07.python-in-wasm"});}finally{methods.MultimodalPipeline.prototype.fit=originalFit;close(producerController);}
  assert.deepEqual(producerController.audit.map(row=>row.operation),["hydrate","PREDICT","dispose","release"]);
  compareNumbers(replayPredictionValues(dag,producerArchive.replay,wire.prediction_sources.nir.sample_ids,wire.target_names),wire.expected_prediction,"unchanged Python archive");
  assert.equal(outcome.selected_variant_id,wire.training_outcome.selected_variant_id);assert.equal(search.selected_trial_index,wire.search_result.selected_trial_index);
  assert.equal(search.trials.length,wire.search_result.trials.length);
  search.trials.forEach((trial,index)=>{assert.deepEqual(trial.params,wire.search_result.trials[index].params);compareNumbers([trial.score],[wire.search_result.trials[index].score],"native HPO score");compareScores(trial.scores,wire.search_result.trials[index].scores);});
  fs.writeFileSync(outputPath,JSON.stringify({host:"wasm",training_outcome:outcome,package:packageValue,archive_inputs:assembled,search_result:search,replay,producer_archive:producerArchive,training_lifecycle:controller.audit,replay_lifecycle:replayController.audit,producer_archive_lifecycle:producerController.audit,runtime:{execution_host:"wasm",signed_controller:controller.controllerId,node:process.version,dag_wasm_sha256:hash(fs.readFileSync(path.join(dagPath,"dag_ml_wasm_bg.wasm"))),methods_dist:path.resolve(methodsPath)}},null,2)+"\n");
}
