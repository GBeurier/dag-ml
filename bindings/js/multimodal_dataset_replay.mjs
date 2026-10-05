/** DAG-owned high-level native U07 replay, no SDK/workspace dependency. */
import {N4mWasmMultimodalController, SOURCE_ORDER, strictJson} from './n4m_multimodal_controller.mjs';
import {replayMultimodalArchive} from './multimodal_archive.mjs';

const stable=value=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,stable(value[key])])):value;
const same=(a,b)=>JSON.stringify(stable(a))===JSON.stringify(stable(b));
const require=(condition,message)=>{if(!condition)throw new Error(message);};

export async function replayMultimodalDatasetArchive({archiveBytes,readPortableArchiveV2,dagMl,methods,current,digest,
  outcomeId='outcome:public.multimodal',runId='run:public.multimodal'}) {
  const archive=await readPortableArchiveV2(archiveBytes),packagePath=archive.manifest.replay.portable_predictor_package.member_path;
  const packageText=new TextDecoder('utf-8',{fatal:true}).decode(archive.members[packagePath]),packageValue=strictJson(packageText);
  dagMl.validate_archive_v2_portable_payloads_json(JSON.stringify(archive.manifest),packageText,
    JSON.stringify(Object.fromEntries(Object.entries(archive.members).map(([name,bytes])=>[name,Array.from(bytes)]))));
  const plan=packageValue.effective_plan,nodes=plan.graph_plan.graph.nodes;
  require(nodes.length===1&&nodes[0].kind==='model'&&nodes[0].operator.type==='N4mMultimodalPipeline','One complete raw multimodal model required');
  const node=nodes[0],selected=plan.node_plans[node.id],owner=selected.controller_id;
  require(['python','wasm','r','octave'].some(host=>owner===`controller:methods.${host}.multimodal`)&&node.metadata?.controller_id===owner,'Unknown signed producer owner');
  require(selected.data_bindings.length===1&&same(selected.data_bindings[0].source_ids,current.source_ids),'Current source binding differs');
  const schemas=node.operator.source_schemas;
  require(same(Object.keys(schemas).sort(),Object.keys(current.source_schemas).sort()),'Current source names differ');
  for(const name of SOURCE_ORDER)require(same({...schemas[name],identity:strictJson(schemas[name].identity)},
    {...current.source_schemas[name],identity:strictJson(current.source_schemas[name].identity)}),`Current source schema differs: ${name}`);
  const bindings=packageValue.output_bindings;
  require(bindings.length===1&&same(bindings[0].target_names,['y']),'One U07 output named y required');
  const relations=current.coordinator_relations,relationFingerprint=dagMl.sample_relation_set_fingerprint_json(JSON.stringify(relations)),envelopes={};
  for(const requirement of packageValue.execution_bundle.data_requirements) {
    const key=`${requirement.node_id}.${requirement.input_name}`;
    require(!(key in envelopes),'Repeated native data requirement');
    envelopes[key]=JSON.parse(dagMl.attach_predict_cohort_to_envelope_json(JSON.stringify({schema_version:1,
      schema_fingerprint:requirement.schema_fingerprint,plan_fingerprint:requirement.plan_fingerprint,
      relation_fingerprint:relationFingerprint,data_content_fingerprint:current.data_content_fingerprint,target_content_fingerprint:null,coordinator_relations:relations}),
      JSON.stringify({role:'inference',relations,target_names:['y'],data_content_fingerprint:current.data_content_fingerprint,target_content_fingerprint:null})));
  }
  require(Object.keys(envelopes).length>0,'Missing replay data requirements');
  const request=JSON.parse(dagMl.sign_training_replay_request_json(JSON.stringify({schema_version:1,request_id:'replay:public.multimodal',
    source_outcome_fingerprint:packageValue.training_outcome.outcome_fingerprint,phase:'PREDICT',data_envelope_keys:Object.keys(envelopes).sort(),
    output_binding_ids:[bindings[0].binding_id],request_fingerprint:'0'.repeat(64)})));
  const positions=Object.fromEntries(SOURCE_ORDER.map(name=>[name,new Map(current.sources[name].sample_ids.map((id,i)=>[id,i]))]));
  const controller=new N4mWasmMultimodalController({methods,operators:{[node.id]:node.operator},nodeParams:{[node.id]:selected.params??{}},
    sourceIds:current.source_ids,targetNames:['y'],digest,allowFit:false,controllerId:owner,resolveTargets:null,
    resolveFeatures:({view})=>{
      const blocks={};
      for(const name of SOURCE_ORDER) {
        const source=current.sources[name],rows=view.sample_ids.map(id=>positions[name].get(id));
        require(rows.every(i=>i!==undefined),'Unknown current source sample ID');
        if(name==='metadata')blocks[name]=rows.map(i=>source.rows[i]);
        else {
          const width=source.shape.slice(1).reduce((a,b)=>a*b,1),Type=schemas[name].dtype==='float32'?Float32Array:Float64Array;
          const data=new Type(rows.length*width);rows.forEach((i,j)=>data.set(source.data.slice(i*width,(i+1)*width),j*width));
          blocks[name]={data,shape:[rows.length,...source.shape.slice(1)]};
        }
      }
      return {sampleIds:[...view.sample_ids],blocks,sourceSchemas:schemas};
    }});
  try {
    const evidence=await replayMultimodalArchive({archiveBytes,readPortableArchiveV2,dagMl,controller,dataEnvelopes:envelopes,request,outcomeId,runId});
    const outputs=evidence.replay.outputs;
    require(outputs.length===1&&outputs[0].predictions.length===1,'One complete native prediction block required');
    const block=outputs[0].predictions[0];
    const alignment=JSON.parse(dagMl.align_named_source_rows_json(JSON.stringify({sample_ids:current.sample_ids,required_source_ids:['prediction'],sources:[{source_id:'prediction',sample_ids:block.sample_ids}]})));
    require(controller.audit.every(event=>!['FIT_CV','REFIT','fit'].includes(event.operation)),'Cold replay unexpectedly performed training');
    return {sample_ids:[...current.sample_ids],target_names:block.target_names,values:alignment.sources[0].row_indices.map(i=>block.values[i]),
      training_performed:false,outcome:evidence.replay,audit:structuredClone(controller.audit),archive_sha256:evidence.archiveSha256};
  } finally {controller.close();}
}
