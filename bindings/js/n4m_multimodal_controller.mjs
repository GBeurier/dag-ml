/** Complete raw U07 early fusion. Only Methods learns/transforms features. */
export const SOURCE_ORDER = Object.freeze(["nir", "image", "series", "metadata"]);
export const MULTIMODAL_SCHEMA = "dagml.methods.multimodal.v1";
const KIND = "methods_multimodal_pipeline", MAX_PAYLOAD = 134217728, MAX_STATE = 67108864;
const PARAMS = ["model__alpha", "source_weights__image", "transformers__image__n_components"];
const requireCondition = (ok, message) => { if (!ok) throw new Error(message); };
const object = value => value !== null && typeof value === "object" && !Array.isArray(value);
const keys = (value, expected) => object(value) && Object.keys(value).sort().join("\0") === [...expected].sort().join("\0");
const canonical = value => Array.isArray(value) ? value.map(canonical) : object(value) ?
  Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
const same = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const strings = value => Array.isArray(value) && value.length > 0 && value.every(x => typeof x === "string" && x.length > 0) && new Set(value).size === value.length;
const boundedText = (value, bound) => typeof value === "string" && new TextEncoder().encode(value).length > 0 && new TextEncoder().encode(value).length <= bound;
// Empty UTF-8 category cells are valid; IDs and identity text remain nonempty.
export const utf8Cell = (value, bound) => typeof value === "string" && new TextEncoder().encode(value).length <= bound;

/** Reject duplicate fields and nonfinite numbers before ordinary JSON decoding. */
export function strictJson(text, rawMember = null) {
  requireCondition(boundedText(text, MAX_PAYLOAD), "Bounded UTF-8 JSON required");
  let i = 0, memberToken;
  const whitespace = () => { while (/\s/.test(text[i] ?? "") && i < text.length) i++; };
  const string = () => {
    const start = i++;
    while (i < text.length) { const c = text[i++]; if (c === "\\") i++; else if (c === '"') return JSON.parse(text.slice(start, i)); }
    throw new Error("Unterminated JSON string");
  };
  const value = depth => {
    requireCondition(depth <= 128, "JSON nesting budget exceeded"); whitespace();
    if (text[i] === '"') { string(); return; }
    if (text[i] === "{" || text[i] === "[") {
      const isObject = text[i++] === "{", end = isObject ? "}" : "]", seen = new Set();
      whitespace(); if (text[i] === end) { i++; return; }
      while (i < text.length) {
        let key;
        if (isObject) { requireCondition(text[i] === '"', "Expected JSON key"); key = string(); requireCondition(!seen.has(key), "Duplicate JSON key"); seen.add(key); whitespace(); requireCondition(text[i++] === ":", "Expected colon"); }
        whitespace(); const first = i; value(depth + 1);
        if (isObject && depth === 0 && key === rawMember) memberToken = text.slice(first, i);
        whitespace(); if (text[i] === end) { i++; return; }
        requireCondition(text[i++] === ",", "Expected comma"); whitespace();
      }
      throw new Error("Incomplete JSON container");
    }
    const match = /^(?:null|true|false|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)/.exec(text.slice(i));
    requireCondition(match !== null, "Invalid JSON value");
    if (!/^(null|true|false)$/.test(match[0])) requireCondition(Number.isFinite(Number(match[0])), "Nonfinite JSON number");
    i += match[0].length;
  };
  value(0); whitespace(); requireCondition(i === text.length, "Trailing JSON data");
  if (rawMember !== null) { requireCondition(typeof rawMember === "string" && memberToken !== undefined, "Missing raw JSON member"); return memberToken; }
  return JSON.parse(text);
}

export function validateRecipe(recipe, schemas) {
  requireCondition(keys(recipe, ["schema_version", "fusion", "source_order", "encoders", "source_weights", "model"]) && recipe.schema_version === 1 && recipe.fusion === "early" && same(recipe.source_order, SOURCE_ORDER), "Closed early-fusion recipe required");
  requireCondition(keys(schemas, SOURCE_ORDER) && keys(recipe.encoders, SOURCE_ORDER) && keys(recipe.source_weights, SOURCE_ORDER), "Complete source recipe required");
  SOURCE_ORDER.forEach((name, index) => {
    const schema = schemas[name];
    requireCondition(keys(schema, ["representation_id", "input_shape", "dtype", "identity"]) && schema.representation_id === ["signal_1d", "rgb_image", "series_mv", "tabular_mixed"][index], "Raw source representation mismatch");
    requireCondition(Array.isArray(schema.input_shape) && schema.input_shape.length > 0 && schema.input_shape.length <= 7 && schema.input_shape.every(size => Number.isSafeInteger(size) && size > 0) && schema.input_shape.reduce((a,b) => a*b,1) <= 1048576 && (name !== "metadata" || same(schema.input_shape, [2])), "Fixed positive raw shape required");
    requireCondition(boundedText(schema.dtype,128) && boundedText(schema.identity,1048576), "Source identity budget exceeded"); strictJson(schema.identity);
    requireCondition(typeof recipe.source_weights[name] === "number" && Number.isFinite(recipe.source_weights[name]) && recipe.source_weights[name] >= 0, "Finite nonnegative source weight required");
  });
  requireCondition(same(recipe.encoders.nir, {kind:"standard_scaler",with_mean:true,with_std:true}) && same(recipe.encoders.metadata,{kind:"column_transformer",numeric_columns:[0],categorical_columns:[1],with_mean:true,with_std:true,handle_unknown:"ignore",sparse_output:false,drop:null}), "Closed native scaler/mixed encoder required");
  for (const name of ["image","series"]) {
    const encoder = recipe.encoders[name];
    requireCondition(keys(encoder,["kind","n_components","whiten","random_state"]) && encoder.kind === "tensor_pca" && Number.isInteger(encoder.n_components) && encoder.n_components > 0 && encoder.n_components <= Math.min(2147483647, schemas[name].input_shape.reduce((a,b)=>a*b,1)) && encoder.whiten === false && Number.isInteger(encoder.random_state) && encoder.random_state >= 0 && encoder.random_state <= 4294967295, "Declared unwhitened PCA recipe required");
  }
  const model = recipe.model;
  requireCondition(keys(model,["method_id","params"]) && model.method_id === "models.regularized.ridge" && keys(model.params,["alpha","center_x","center_y","scale_x"]) && typeof model.params.alpha === "number" && Number.isFinite(model.params.alpha) && model.params.alpha >= 0 && model.params.center_x === true && model.params.center_y === true && model.params.scale_x === false, "Closed native Ridge recipe required");
}

export function recipeForNode(operator, params) {
  requireCondition(keys(operator,["type","recipe","source_schemas"]) && operator.type === "N4mMultimodalPipeline" && object(params) && Object.keys(params).every(key=>PARAMS.includes(key)), "Explicit multimodal operator/effective parameters required");
  const recipe = structuredClone(operator.recipe);
  if ("model__alpha" in params) recipe.model.params.alpha = params.model__alpha;
  if ("source_weights__image" in params) recipe.source_weights.image = params.source_weights__image;
  if ("transformers__image__n_components" in params) recipe.encoders.image.n_components = params.transformers__image__n_components;
  validateRecipe(recipe, operator.source_schemas); return recipe;
}

function validateWrapper(saved) {
  requireCondition(keys(saved,["schema","node_id","params_fingerprint","target_names","recipe","source_schemas","state"]) && saved.schema === MULTIMODAL_SCHEMA && strings(saved.target_names) && saved.target_names.length === 1 && boundedText(saved.target_names[0],4096), "Closed complete predictor wrapper required");
  requireCondition(Array.isArray(saved.state) && saved.state.length >= 28 && saved.state.length <= MAX_STATE && saved.state.every(byte=>Number.isInteger(byte)&&byte>=0&&byte<=255), "Exact bounded native state bytes required");
  requireCondition(same(saved.state.slice(0,12),[78,52,77,70,1,0,0,0,2,0,0,0]), "Unsupported N4MF header"); validateRecipe(saved.recipe,saved.source_schemas);
}

export function multimodalManifest(host="wasm") {
  requireCondition(["python","wasm","r","octave"].includes(host), "Unknown host");
  return {controller_id:`controller:methods.${host}.multimodal`,controller_version:"1.0.0",operator_kind:"model",priority:0,supported_phases:["FIT_CV","REFIT","PREDICT"],
    input_ports:[{name:"x",kind:"data",representation:"feature_block_set",cardinality:"one",description:""}],
    output_ports:[{name:"y_hat",kind:"prediction",representation:null,cardinality:"one",description:""},{name:"model",kind:"artifact",representation:null,cardinality:"one",description:""}],
    data_requirements:{schema_version:1,default_fusion:{mode:"dict_by_source",alignment:"sample_id",adapter_id:null,params:{}},metadata:{},ports:[{name:"x",accepted_representations:["feature_block_set"],accepted_types:["multi_block"],rank:null,multi_source:true,optional:false,metadata:{}}]},
    capabilities:["deterministic","thread_safe","process_safe","emits_predictions","emits_artifacts","stateful","uses_core_rng"],fit_scope:"fold_train",rng_policy:"externally_deterministic",artifact_policy:"serializable"};
}

/** Resolver supplies raw tensors/cells in exact native task sample order. */
export class N4mWasmMultimodalController {
  constructor({methods,operators,resolveFeatures,resolveTargets=null,targetNames=["y"],sourceIds=SOURCE_ORDER,nodeParams={},digest,allowFit=true,controllerId="controller:methods.wasm.multimodal"}) {
    requireCondition(methods?.MultimodalPipeline && object(operators) && Object.keys(operators).length && typeof resolveFeatures === "function" && typeof digest === "function" && typeof allowFit === "boolean" && allowFit === (typeof resolveTargets === "function"), "Methods, raw resolvers and exact replay policy required");
    requireCondition(strings(targetNames)&&targetNames.length===1&&strings(sourceIds)&&sourceIds.length===4, "One target and four ordered wire sources required");
    this.methods=methods; this.operators=structuredClone(operators); this.nodeParams=structuredClone(nodeParams); this.resolveFeatures=resolveFeatures; this.resolveTargets=resolveTargets;
    const ownerHost=["python","wasm","r","octave"].find(host=>controllerId===`controller:methods.${host}.multimodal`);
    requireCondition(ownerHost&&(!allowFit||ownerHost==="wasm"),"Closed producer owner required; fitting requires WASM ownership");
    this.ownerHost=ownerHost;this.executionHost="wasm";this.targetNames=[...targetNames]; this.sourceIds=[...sourceIds]; this.digest=digest; this.allowFit=allowFit; this.controllerId=controllerId; this.plugin=`dagml.methods.${ownerHost}.multimodal`;
    this.models=new Map(); this.artifacts=new Map(); this.nextHandle=1; this.closed=false; this.audit=[]; this.callback=this.invoke.bind(this);
    for(const [node,operator] of Object.entries(this.operators)) recipeForNode(operator,this.nodeParams[node]??{});
    requireCondition(Object.keys(nodeParams).every(node=>node in operators),"Foreign expected parameter node");
  }
  manifest() { return multimodalManifest(this.ownerHost); }
  _hash(bytes) { const value=this.digest(bytes); requireCondition(typeof value==="string"&&/^[0-9a-f]{64}$/.test(value),"Synchronous SHA-256 required");return value; }
  _keep(entry) { requireCondition(Number.isSafeInteger(this.nextHandle),"Handle space exhausted"); const handle=this.nextHandle++;this.models.set(handle,entry);return {handle,kind:"model",owner_controller:this.controllerId}; }
  _features(task,partition) {
    const views=Object.entries(task.data_views??{}).filter(([,view])=>view.partition===partition); requireCondition(views.length===1,"One native raw multimodal view required");
    const [key,view]=views[0]; requireCondition(strings(view.sample_ids)&&same(view.source_ids,this.sourceIds)&&!view.include_augmented&&(["fold_validation","predict"].includes(partition)||!view.include_excluded)&&!(view.columns??[]).length,"Raw native view scope mismatch");
    const resolved=this.resolveFeatures({key,view:structuredClone(view),task:structuredClone(task)});
    requireCondition(resolved&&!resolved.then&&same(resolved.sampleIds,view.sample_ids)&&keys(resolved.blocks,SOURCE_ORDER)&&same(resolved.sourceSchemas,this.operators[task.node_plan.node_id].source_schemas),"Current independently resolved raw schema/sample order mismatch");
    for(const name of SOURCE_ORDER) {
      const block=resolved.blocks[name], shape=this.operators[task.node_plan.node_id].source_schemas[name].input_shape;
      if(name==="metadata") requireCondition(Array.isArray(block)&&block.length===view.sample_ids.length&&block.every(row=>Array.isArray(row)&&row.length===2&&(typeof row[0]==="number"||typeof row[0]==="string")&&utf8Cell(row[1],1048576)),"Raw mixed UTF-8 rows required");
      else requireCondition(block&&(block.data instanceof Float32Array||block.data instanceof Float64Array)&&same(block.shape,[view.sample_ids.length,...shape])&&block.shape.reduce((a,b)=>a*b,1)<=16777216&&block.data.every(Number.isFinite),"Raw finite tensor shape/budget mismatch");
    }
    return resolved;
  }
  _targets(sampleIds,task) {
    requireCondition(this.resolveTargets!==null,"Replay cannot read training targets");const value=this.resolveTargets({sampleIds:[...sampleIds],task:structuredClone(task)});
    requireCondition(value&&!value.then&&same(value.sampleIds,sampleIds)&&value.matrix?.data instanceof Float64Array&&value.matrix.rows===sampleIds.length&&value.matrix.cols===1&&value.matrix.data.length===sampleIds.length&&value.matrix.data.every(Number.isFinite),"Aligned finite scalar targets required");return value.matrix;
  }
  _result(task,resolved,model,artifacts=[]) {
    const matrix=model.predict(resolved.blocks,resolved.sourceSchemas), values=matrix.data??matrix;
    requireCondition((values instanceof Float32Array||values instanceof Float64Array||Array.isArray(values))&&values.length===resolved.sampleIds.length&&Array.from(values).every(Number.isFinite),"Native scalar prediction width mismatch");
    const node=task.node_plan, rows=Array.from(values,value=>[value]);
    const result={node_id:node.node_id,outputs:{},artifacts,artifact_handles:{},predictions:[{producer_node:node.node_id,partition:task.phase==="FIT_CV"?"validation":"final",fold_id:task.fold_id??null,sample_ids:resolved.sampleIds,values:rows,target_names:this.targetNames}],
      lineage:{record_id:["lineage:methods-multimodal",task.run_id,node.node_id,task.phase,task.variant_id??"base",task.fold_id??"full"].join(":"),run_id:task.run_id,node_id:node.node_id,phase:task.phase,controller_id:this.controllerId,controller_version:"1.0.0",variant_id:task.variant_id??null,fold_id:task.fold_id??null,branch_path:task.branch_path??[],input_lineage:[],artifact_refs:artifacts,params_fingerprint:node.params_fingerprint,data_model_shape_fingerprint:null,aggregation_policy_fingerprint:null,seed:task.seed,unsafe_flags:[],metrics:{},loss_attestations:[],early_stopping_records:[]}};
    if(this.resolveTargets!==null) result.regression_targets=[{level:"sample",unit_ids:resolved.sampleIds.map(id=>({level:"sample",id})),values:Array.from(this._targets(resolved.sampleIds,task).data,value=>[value]),target_names:this.targetNames}];
    if(["FIT_CV","REFIT"].includes(task.phase)&&Object.values(task.data_views??{}).some(view=>view.partition==="predict")) {
      const test=this._features(task,"predict"),predicted=model.predict(test.blocks,test.sourceSchemas),testValues=predicted.data??predicted;
      requireCondition(testValues.length===test.sampleIds.length&&Array.from(testValues).every(Number.isFinite),"Native test prediction width mismatch");
      result.predictions.push({producer_node:node.node_id,partition:"test",fold_id:task.fold_id??null,sample_ids:test.sampleIds,values:Array.from(testValues,value=>[value]),target_names:this.targetNames});
      if(this.resolveTargets!==null)result.regression_targets.push({level:"sample",unit_ids:test.sampleIds.map(id=>({level:"sample",id})),values:Array.from(this._targets(test.sampleIds,task).data,value=>[value]),target_names:this.targetNames});
    }
    this.audit.push({operation:task.phase,sample_ids:resolved.sampleIds,node:node.node_id});return result;
  }
  artifactPayload(id) {requireCondition(!this.closed&&this.artifacts.has(id),"Unknown complete predictor");return this.artifacts.get(id).slice();}
  hydrate(request,payload) {
    requireCondition(!this.closed&&payload instanceof Uint8Array&&payload.length<=MAX_PAYLOAD,"Bounded RAW payload required");const ref=request.artifact,hash=this._hash(payload);
    requireCondition(request.controller_id===this.controllerId&&ref.controller_id===this.controllerId&&ref.kind===KIND&&ref.backend==="raw"&&ref.plugin===this.plugin&&ref.plugin_version==="1.0.0"&&ref.native_predictor_descriptor==null&&ref.native_estimator_descriptor==null&&ref.uri===`artifacts/${hash}.json`&&ref.content_fingerprint===hash&&ref.size_bytes===payload.length,"RAW owner, plugin, SHA, URI or size mismatch");
    const saved=strictJson(new TextDecoder("utf-8",{fatal:true}).decode(payload));validateWrapper(saved);
    const operator=this.operators[saved.node_id];requireCondition(operator&&saved.node_id===request.node_id&&saved.params_fingerprint===request.params_fingerprint&&same(saved.target_names,this.targetNames)&&same(saved.source_schemas,operator.source_schemas)&&same(saved.recipe,recipeForNode(operator,this.nodeParams[saved.node_id]??{})),"Selected recipe, target, source schema or node mismatch before native hydration");
    const model=this.methods.MultimodalPipeline.fromState(Uint8Array.from(saved.state),saved.recipe,saved.source_schemas);
    try {const handle=this._keep({model,saved,artifact:structuredClone(ref)});this.audit.push({operation:"hydrate"});return handle;} catch(error){model.dispose();throw error;}
  }
  release(handle) {requireCondition(handle?.kind==="model"&&handle.owner_controller===this.controllerId&&this.models.has(handle.handle),"Unknown/foreign model handle");const entry=this.models.get(handle.handle);this.models.delete(handle.handle);entry.model.dispose();this.audit.push({operation:"dispose"},{operation:"release"});}
  invoke(controllerId,taskJson) {
    requireCondition(!this.closed&&controllerId===this.controllerId,"Wrong or closed controller");const task=strictJson(taskJson);
    // Preserve the original u64 JSON token: JS Numbers cannot represent all seeds.
    const seedToken=task.operation?null:strictJson(taskJson,"seed");
    if(!task.operation)requireCondition(seedToken==="null"||(/^(0|[1-9]\d*)$/.test(seedToken)&&BigInt(seedToken)<=18446744073709551615n),"Exact native optional u64 seed token required");
    let result;
    if(task.operation) {
      requireCondition(task.schema_version===1,"Unsupported bridge schema");
      if(task.operation==="export_artifact_payload")result={operation:"exported_artifact_payload",schema_version:1,payload:Array.from(this.artifactPayload(task.artifact_id))};
      else if(task.operation==="hydrate_artifact_payload") {requireCondition(Array.isArray(task.payload)&&task.payload.length<=MAX_PAYLOAD&&task.payload.every(byte=>Number.isInteger(byte)&&byte>=0&&byte<=255),"Exact bounded payload bytes required");result={operation:"hydrated_artifact_payload",schema_version:1,handle:this.hydrate(task.request,Uint8Array.from(task.payload))};}
      else {requireCondition(task.operation==="release_hydrated_artifact_payload","Unknown artifact operation");this.release(task.handle);result={operation:"released_hydrated_artifact_payload",schema_version:1};}
    } else {
      const node=task.node_plan,operator=this.operators[node.node_id];requireCondition(node.kind==="model"&&node.controller_id===this.controllerId&&node.controller_version==="1.0.0"&&operator&&["FIT_CV","REFIT","PREDICT"].includes(task.phase),"Foreign node owner/phase");
      requireCondition(!Object.keys(task.prediction_inputs??{}).length&&!Object.keys(task.data_view_receipts??{}).length&&!(task.required_loss_attestations??[]).length&&!task.residual_targets&&(!task.fit_influence||(task.fit_influence.mechanism==="uniform_rows"&&!(task.fit_influence.row_weights??[]).length)),"Generated/OOF/residual/loss/nonuniform inputs unsupported");
      const recipe=recipeForNode(operator,Object.hasOwn(node,"params")?node.params:{});
      if(task.phase==="PREDICT") {
        const inputs=Object.entries(task.artifact_inputs??{});requireCondition(inputs.length===1,"One complete predictor required");const [key,input]=inputs[0],handle=task.input_handles[key],entry=this.models.get(handle?.handle);
        requireCondition(handle?.kind==="model"&&handle.owner_controller===this.controllerId&&entry&&input.node_id===node.node_id&&input.controller_id===this.controllerId&&input.params_fingerprint===node.params_fingerprint&&same(input.artifact,entry.artifact)&&entry.saved.params_fingerprint===node.params_fingerprint&&entry.saved.node_id===node.node_id&&same(entry.saved.recipe,recipe)&&same(entry.saved.source_schemas,operator.source_schemas),"PREDICT binding or schema mismatch");result=this._result(task,this._features(task,"predict"),entry.model);
      } else {
        requireCondition(this.allowFit,"Fitting disabled for replay");const train=this._features(task,task.phase==="FIT_CV"?"fold_train":"full_train"),valid=task.phase==="FIT_CV"?this._features(task,"fold_validation"):train;
        requireCondition(task.phase!=="FIT_CV"||train.sampleIds.every(id=>!valid.sampleIds.includes(id)),"Training/validation overlap");
        const model=new this.methods.MultimodalPipeline(recipe,operator.source_schemas);let retained=false;
        try {model.fit(train.blocks,this._targets(train.sampleIds,task));this.audit.push({operation:"fit",sample_ids:train.sampleIds,fold:task.fold_id??null});
          if(task.phase==="FIT_CV")result=this._result(task,valid,model);
          else {const saved={schema:MULTIMODAL_SCHEMA,node_id:node.node_id,params_fingerprint:node.params_fingerprint,target_names:this.targetNames,recipe,source_schemas:operator.source_schemas,state:Array.from(model.exportState())};validateWrapper(saved);const payload=new TextEncoder().encode(JSON.stringify(saved));requireCondition(payload.length<=MAX_PAYLOAD,"Complete predictor budget exceeded");const hash=this._hash(payload),id=["artifact:methods.multimodal",task.run_id,node.node_id,task.variant_id??"base","refit"].join(":");requireCondition(!this.artifacts.has(id),"Duplicate REFIT artifact");const artifact={id,kind:KIND,controller_id:this.controllerId,backend:"raw",uri:`artifacts/${hash}.json`,content_fingerprint:hash,size_bytes:payload.length,plugin:this.plugin,plugin_version:"1.0.0"};result=this._result(task,valid,model,[artifact]);result.artifact_handles[id]=this._keep({model,saved,artifact});this.artifacts.set(id,payload);retained=true;}
        } finally {if(!retained){model.dispose();this.audit.push({operation:"dispose"});}}
      }
    }
    if(seedToken) result.lineage.seed="__DAGML_U64_SEED__";
    return JSON.stringify(result).replace('"seed":"__DAGML_U64_SEED__"',`"seed":${seedToken}`);
  }
  close() {if(this.closed)return;let failure;for(const entry of this.models.values()){try{entry.model.dispose();this.audit.push({operation:"dispose"});}catch(error){failure??=error;}}this.models.clear();this.artifacts.clear();this.closed=true;if(failure)throw failure;}
}
