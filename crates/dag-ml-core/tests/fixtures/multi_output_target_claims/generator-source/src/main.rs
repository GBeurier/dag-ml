include!("pattern.rs");
use std::path::PathBuf;
use std::io::Write;
fn out() -> PathBuf { PathBuf::from(std::env::var("PROBE_OUTPUT").expect("PROBE_OUTPUT")) }
fn save(name:&str, value:&impl serde::Serialize) {
 let bytes=serde_json::to_vec_pretty(value).unwrap();let path=out().join(name);
 if path.exists(){assert_eq!(std::fs::read(path).unwrap(),bytes,"existing scratch input must remain exact");return;}
 let mut f=std::fs::OpenOptions::new().write(true).create_new(true).open(path).unwrap();f.write_all(&bytes).unwrap();
}
fn log_event(v:serde_json::Value) { let mut f=std::fs::OpenOptions::new().create(true).append(true).open(out().join("callbacks.jsonl")).unwrap(); writeln!(f,"{}",v).unwrap(); }
fn stage<T>(label:&str,r:Result<T>)->Result<T> {log_event(serde_json::json!({"event":"stage","label":label,"result":r.as_ref().map(|_|"ok").map_err(|e|e.to_string())}));r}
struct Provider { identities:BTreeMap<String,TrainingDataIdentity>, relations:SampleRelationSet, cohorts:BTreeMap<String,PredictCohort>, next:AtomicU64 }
impl Provider {fn h(&self,k:HandleKind)->HandleRef { HandleRef {handle:self.next.fetch_add(1,Ordering::SeqCst)+10000,kind:k,owner_controller:ControllerId::new("controller:probe.provider").unwrap()} }}
impl RuntimeDataProvider for Provider {
 fn materialize(&self,r:&DataMaterializationRequest)->Result<HandleRef>{log_event(serde_json::json!({"event":"materialize","request":r}));Ok(self.h(HandleKind::Data))}
 fn make_view(&self,r:&DataViewRequest)->Result<HandleRef>{log_event(serde_json::json!({"event":"make_view","request":r}));Ok(self.h(HandleKind::DataView))}
 fn training_data_identity(&self,b:&DataBinding)->Result<Option<TrainingDataIdentity>>{Ok(self.identities.get(&data_binding_requirement_key(&b.node_id,&b.input_name)).cloned())}
 fn coordinator_relations(&self,_:&DataBinding)->Result<Option<SampleRelationSet>> {Ok(Some(self.relations.clone()))}
 fn predict_cohort(&self,b:&DataBinding,_:Phase)->Result<Option<PredictCohort>> {Ok(self.cohorts.get(&data_binding_requirement_key(&b.node_id,&b.input_name)).cloned())}
}
const RAW:&[u8]=b"synthetic-host-fitted-state-v1";
struct RawController {inner:TrainingController, bad_names:bool, hydrated:Mutex<BTreeSet<u64>>}
impl RuntimeController for RawController {
 fn controller_id(&self)->&ControllerId{self.inner.controller_id()}
 fn invoke(&self,t:&NodeTask)->Result<NodeResult>{
  if matches!(t.phase,Phase::Predict|Phase::Explain) {assert!(!t.artifact_inputs.is_empty(),"replay must receive actual saved fitted state");
    assert!(t.input_handles.values().any(|h|h.owner_controller==self.inner.id && self.hydrated.lock().unwrap().contains(&h.handle)),"invoke consumes an actually hydrated saved host state");}
  let mut r=if t.node_plan.node_id.as_str()=="model:second" {
    let second=TrainingController{id:self.inner.id.clone(),state:self.inner.state.clone(),emits_predictions:true,emits_artifact:true,prediction_name:t.node_plan.params.get("toy_target").and_then(|v|v.as_str()).unwrap_or("fat").into()};second.invoke(t)?
  }else{self.inner.invoke(t)?};
  // Extend only the authorized synthetic host result; no saved payload is changed.
  let targets:Vec<String>=serde_json::from_value(t.node_plan.params["toy_targets"].clone()).unwrap();
  for block in &mut r.predictions {block.target_names=targets.clone();block.values=vec![vec![0.0;targets.len()];block.sample_ids.len()];}
  for block in &mut r.regression_targets {block.target_names=targets.clone();block.values=vec![vec![0.0;targets.len()];block.unit_ids.len()];}
  if t.phase==Phase::Explain {
   r.explanations=vec![ExplanationBlock{producer_node:t.node_plan.node_id.clone(),producer_port:Some("oof".into()),method:"synthetic_zero_importance".into(),target_name:Some(targets[0].clone()),payload:serde_json::json!({"importance":0.0})}];
  }
  for a in &mut r.artifacts {a.backend=Some(ArtifactBackend::Raw);a.content_fingerprint=Some(format!("{:x}",Sha256::digest(RAW)));a.uri=Some("artifacts/toy-state.bin".into());a.size_bytes=Some(RAW.len() as u64);}
  r.lineage.artifact_refs=r.artifacts.clone();
  if self.bad_names && t.phase==Phase::Predict {for b in &mut r.predictions {b.target_names=(0..b.target_names.len()).map(|i|format!("late-wrong-{i}")).collect();}}
  log_event(serde_json::json!({"event":"raw_result","task":t,"result":r})); Ok(r)
 }
 fn export_artifact_payload(&self,id:&ArtifactId)->Result<Option<Vec<u8>>>{log_event(serde_json::json!({"event":"export","id":id}));Ok(Some(RAW.to_vec()))}
 fn hydrate_artifact_payload(&self,r:&ArtifactMaterializationRequest,p:&[u8])->Result<HandleRef>{assert_eq!(p,RAW);log_event(serde_json::json!({"event":"hydrate","request":r,"payload_sha256":format!("{:x}",Sha256::digest(p))}));let h=HandleRef{handle:self.inner.state.handle(),kind:HandleKind::Artifact,owner_controller:self.inner.id.clone()};self.hydrated.lock().unwrap().insert(h.handle);Ok(h)}
 fn release_hydrated_artifact_payload(&self,h:&HandleRef)->Result<()>{log_event(serde_json::json!({"event":"release","handle":h}));assert!(self.hydrated.lock().unwrap().remove(&h.handle));Ok(())}
}
fn registry(f:&Fixture,state:&Arc<CallState>,bad_names:bool)->RuntimeControllerRegistry{
 let mut r=RuntimeControllerRegistry::new();
 r.register(Box::new(TrainingController{id:ControllerId::new("controller:transform.mock").unwrap(),state:state.clone(),emits_predictions:false,emits_artifact:false,prediction_name:"aux".into()})).unwrap();
 r.register(Box::new(RawController{inner:TrainingController{id:ControllerId::new("controller:model.mock").unwrap(),state:state.clone(),emits_predictions:true,emits_artifact:true,prediction_name:"protein".into()},bad_names,hydrated:Mutex::new(BTreeSet::new())})).unwrap();
 let mut trusted=ControllerRegistry::new();for m in &f.request.controller_manifests{trusted.register(m.clone()).unwrap();}
 stage("validate_runtime_controller_manifests",validate_runtime_controller_manifests(&f.request.project().unwrap().plan,&trusted)).unwrap();r
}
fn train(mode:&str)->Result<(Fixture,TrainingOutcome,PortablePredictorPackage)> {
 let mut f=fixture(true,false);add_model_probability_port(&mut f);
 let second=if mode.ends_with("equal") {vec!["protein".to_string()]} else {vec!["fat".to_string()]};
 f.request.graph.nodes.iter_mut().find(|n|n.id.as_str()=="model:base").unwrap().params.insert("second_targets".into(),serde_json::json!(second));
 f.request.options.outputs[0].output_id="output:a".into();
 let mut b=f.request.options.outputs[0].clone();b.output_id="output:b".into();b.port_name=Some("probability".into());b.target_names=second;
 f.request.options.outputs.push(b);f.request.options.selection_output_id="output:a".into();
 if mode.starts_with("models-") {
   // Public two-model topology. The one actual ancestor binding is moved to the transform.
   let second_id=node("model:second");
   let base=f.request.graph.nodes.iter_mut().find(|n|n.id.as_str()=="model:base").unwrap();
   base.ports.outputs.truncate(1);base.params.remove("second_targets");
   let mut sibling=base.clone();sibling.id=second_id.clone();sibling.seed_label=Some("second".into());sibling.params.insert("toy_target".into(),serde_json::json!(f.request.options.outputs[1].target_names[0]));
   f.request.graph.nodes.push(sibling);
   let mut edge=f.request.graph.edges[0].clone();edge.target.node_id=second_id.clone();f.request.graph.edges.push(edge);
   let mut shape=f.request.campaign.shape_plans[&node("model:base")].clone();shape.node_id=second_id.clone();f.request.campaign.shape_plans.insert(second_id.clone(),shape);
   f.request.options.outputs[1].node_id=second_id;f.request.options.outputs[1].port_name=Some("oof".into());
   f.request.controller_manifests.iter_mut().find(|m|m.controller_id.as_str()=="controller:model.mock").unwrap().output_ports.truncate(1);
   let mut bindings=f.request.campaign.data_bindings.remove(&node("model:base")).unwrap();
   bindings[0].node_id=node("transform:snv");f.request.campaign.data_bindings.insert(node("transform:snv"),bindings);
   if mode.ends_with("distinct") {
     let mut other=f.request.graph.nodes.iter().find(|n|n.id.as_str()=="transform:snv").unwrap().clone();other.id=node("transform:second");other.seed_label=Some("second-input".into());f.request.graph.nodes.push(other);
     f.request.graph.edges[1].source.node_id=node("transform:second");
     let mut shape=f.request.campaign.shape_plans[&node("transform:snv")].clone();shape.node_id=node("transform:second");f.request.campaign.shape_plans.insert(node("transform:second"),shape);
     let mut bindings=f.request.campaign.data_bindings[&node("transform:snv")].clone();bindings[0].node_id=node("transform:second");f.request.campaign.data_bindings.insert(node("transform:second"),bindings);
   }
   f.request.data_identities=f.request.campaign.data_bindings.values().flatten().map(|b| {
     let envelope=ExternalDataPlanEnvelope{schema_version:1,schema_fingerprint:b.schema_fingerprint.clone(),plan_fingerprint:b.plan_fingerprint.clone(),relation_fingerprint:Some(f.relations.fingerprint().unwrap()),data_content_fingerprint:Some("a".repeat(64)),target_content_fingerprint:Some("b".repeat(64)),coordinator_relations:Some(f.relations.clone()),predict_cohort:None};
     TrainingDataIdentity::from_binding_envelope(b,&envelope).unwrap()
   }).collect();f.request.data_identities.sort_by(|a,b|a.requirement_key.cmp(&b.requirement_key));
 }
 let (a_names,b_names): (Vec<&str>,Vec<&str>)=match mode {
   "models-shared" => (vec!["protein"],vec!["fat"]),
   "models-multi-equal" => (vec!["protein","fat"],vec!["protein","fat"]),
   "models-multi-reversed" | "models-multi-reversed-distinct" => (vec!["protein","fat"],vec!["fat","protein"]),
   "models-multi-overlap" => (vec!["protein","fat"],vec!["fat","water"]),
   _ => panic!("unknown bounded mode"),
 };
 for (i,names) in [a_names,b_names].into_iter().enumerate() {
   f.request.options.outputs[i].target_names=names.iter().map(|s|s.to_string()).collect();
   f.request.options.outputs[i].target_units=vec![Some("percent".into());names.len()];
   f.request.options.outputs[i].class_labels=vec![vec![];names.len()];
   let id=f.request.options.outputs[i].node_id.clone();
   f.request.graph.nodes.iter_mut().find(|n|n.id==id).unwrap().params.insert("toy_targets".into(),serde_json::json!(names));
 }
 // Single concrete variant is enough; preserve owner-generated variant and fold identities.
 f.request.campaign.generation.dimensions[0].choices.truncate(1);
 let x=serde_json::json!({"sample_ids":["sample:1","sample:2","sample:3","sample:4"],"x":[[1.0],[2.0],[3.0],[4.0]]});
 let y=serde_json::json!({"sample_ids":["sample:1","sample:2","sample:3","sample:4"],"target_names":["protein","fat","water"],"y":vec![vec![0.0;3];4]});
 save("host-training-X.json",&x);save("host-training-y.json",&y);
 let data_hash=format!("{:x}",Sha256::digest(std::fs::read(out().join("host-training-X.json")).unwrap()));
 let target_hash=format!("{:x}",Sha256::digest(std::fs::read(out().join("host-training-y.json")).unwrap()));
 f.request.data_identities=f.request.campaign.data_bindings.values().flatten().map(|b| {
   let e=ExternalDataPlanEnvelope{schema_version:1,schema_fingerprint:b.schema_fingerprint.clone(),plan_fingerprint:b.plan_fingerprint.clone(),relation_fingerprint:Some(f.relations.fingerprint().unwrap()),data_content_fingerprint:Some(data_hash.clone()),target_content_fingerprint:Some(target_hash.clone()),coordinator_relations:Some(f.relations.clone()),predict_cohort:None};
   TrainingDataIdentity::from_binding_envelope(b,&e).unwrap()
 }).collect();f.request.data_identities.sort_by(|a,b|a.requirement_key.cmp(&b.requirement_key));
 for m in &mut f.request.controller_manifests{if !m.supported_phases.contains(&Phase::Explain){m.supported_phases.insert(Phase::Explain);}}
 f.request.request_fingerprint=f.request.compute_fingerprint()?;
 save("training-request.json",&f.request);save("training-relations.json",&f.relations);
 stage("request.validate",f.request.validate())?;
 let projection=stage("request.project",f.request.project())?;stage("projection.validate",projection.validate())?;
 f.preferred=projection.plan.variants[0].variant_id.clone();
 f.influence=stage("TrainingInfluenceManifest::derive_for_projection",TrainingInfluenceManifest::derive_for_projection(&projection,&f.request,&f.relations))?;
 save("projection.json",&projection);save("training-influence.json",&f.influence);
 let state=Arc::new(CallState::default());*state.preferred.lock().unwrap()=Some(f.preferred.clone());*state.emit_explicit_model_ports.lock().unwrap()=true;
 let controllers=registry(&f,&state,false);
 let provider=Provider{identities:f.request.data_identities.iter().map(|i|(i.requirement_key.clone(),i.clone())).collect(),relations:f.relations.clone(),cohorts:BTreeMap::new(),next:AtomicU64::new(0)};
 let mut store=InMemoryArtifactStore::new();
 let outcome=stage("execute_training",execute_training(TrainingExecutionInput{request:&f.request,outcome_id:format!("outcome:probe.{mode}"),run_id:RunId::new(format!("run:probe.{mode}")).unwrap(),bundle_id:BundleId::new(format!("bundle:probe.{mode}")).unwrap(),controllers:&controllers,data_provider:&provider,relations:&f.relations,training_influence:&f.influence,artifact_store:&mut store,warnings:vec![],diagnostics:BTreeMap::new()}))?;
 stage("outcome.validate",outcome.validate())?;save("training-outcome.json",&outcome);
 let package=stage("to_portable_predictor_package",outcome.to_portable_predictor_package(format!("package:probe.{mode}"),FittedArtifactMode::PortableRequired,ArtifactLoadMode::NativePortable))?;
 stage("package.validate",package.validate())?;save("package.json",&package);
 Ok((f,outcome,package))
}
fn replay(f:&Fixture,o:&TrainingOutcome,p:&PortablePredictorPackage,label:&str,selected:Vec<String>,claim:Option<Vec<String>>,bad:bool)->serde_json::Value {
 let rel=SampleRelationSet{records:(11..=14).map(|i|SampleRelation::new(ObservationId::new(format!("observation:{i}")).unwrap(),sample(&format!("sample:{i}")))).collect()};
 let x=serde_json::json!({"sample_ids":["sample:11","sample:12","sample:13","sample:14"],"x":[[11.0],[12.0],[13.0],[14.0]]});
 save("host-replay-X.json",&x);
 let xhash=format!("{:x}",Sha256::digest(std::fs::read(out().join("host-replay-X.json")).unwrap()));
 let y=serde_json::json!({"sample_ids":["sample:11","sample:12","sample:13","sample:14"],"target_names":["protein","fat","water"],"y":vec![vec![0.0;3];4]});
 save("host-legacy-replay-y.json",&y);
 let legacy_y_hash=format!("{:x}",Sha256::digest(std::fs::read(out().join("host-legacy-replay-y.json")).unwrap()));
 let cohort=claim.clone().map(|names|PredictCohort::from_relations(PredictCohortRole::Inference,rel.clone(),names,xhash.clone(),None).unwrap());
 let mut envelopes:BTreeMap<_,_>=o.execution_bundle.data_requirements.iter().map(|d|(d.key(),ExternalDataPlanEnvelope{schema_version:if cohort.is_some(){2}else{1},schema_fingerprint:d.schema_fingerprint.clone(),plan_fingerprint:d.plan_fingerprint.clone(),relation_fingerprint:Some(rel.fingerprint().unwrap()),coordinator_relations:Some(rel.clone()),data_content_fingerprint:Some(xhash.clone()),target_content_fingerprint:if cohort.is_some(){None}else{Some(legacy_y_hash.clone())},predict_cohort:cohort.clone()})).collect();
 if label=="both-per-key" {
   for (key,e) in &mut envelopes {e.predict_cohort=Some(PredictCohort::from_relations(PredictCohortRole::Inference,rel.clone(),p.output_bindings.iter().find(|b| if key=="transform:snv.x"{b.binding_id=="output:a"}else{b.binding_id=="output:b"}).unwrap().target_names.clone(),xhash.clone(),None).unwrap());}
 }
 for e in envelopes.values(){e.validate().unwrap();}
 let mut req=TrainingReplayRequest{schema_version:TRAINING_REPLAY_REQUEST_SCHEMA_VERSION,request_id:format!("replay:{label}"),source_outcome_fingerprint:o.outcome_fingerprint.clone(),phase:Phase::Predict,data_envelope_keys:envelopes.keys().cloned().collect(),output_binding_ids:selected,request_fingerprint:"0".repeat(64)};
 req.request_fingerprint=req.compute_fingerprint().unwrap();
 let request_validation=req.validate().map_err(|e|e.to_string());
 save(&format!("{label}-request.json"),&req);save(&format!("{label}-envelopes.json"),&envelopes);
 log_event(serde_json::json!({"event":"case_start","label":label}));
 let preflight=preflight_loaded_predictor_replay(p,&req,&envelopes).map_err(|e|e.to_string());
 let loaded=LoadedPredictor::<HandleRef>::new(p.clone(),BTreeMap::new()).unwrap();
 let state=Arc::new(CallState::default());*state.emit_explicit_model_ports.lock().unwrap()=true;*state.predict_sample_ids.lock().unwrap()=Some(rel.records.iter().map(|r|r.sample_id.clone()).collect());
 let controllers=registry(f,&state,bad);
 let provider=Provider{identities:BTreeMap::new(),relations:rel,cohorts:envelopes.iter().filter_map(|(k,e)|e.predict_cohort.clone().map(|c|(k.clone(),c))).collect(),next:AtomicU64::new(0)};
 let result=execute_loaded_predictor_replay(LoadedPredictorReplayInput{predictor:&loaded,request:&req,outcome_id:format!("outcome:{label}"),run_id:RunId::new(format!("run:{label}")).unwrap(),controllers:&controllers,data_provider:&provider,data_envelopes:&envelopes,warnings:vec![],diagnostics:BTreeMap::new()});
 let execution=match result{Ok(v)=>{v.validate_against_package(p,&req).unwrap();save(&format!("{label}-replay.json"),&v);serde_json::json!({"ok":true,"output_targets":v.outputs.iter().map(|o|(&o.binding.binding_id,&o.binding.target_names)).collect::<Vec<_>>()})},Err(e)=>serde_json::json!({"ok":false,"error":e.to_string()})};
 let summary=serde_json::json!({"label":label,"selected":req.output_binding_ids,"claim":claim,"request_validation":request_validation,"actual_envelope_claims":envelopes.iter().map(|(k,e)|(k,e.predict_cohort.as_ref().map(|c|c.target_names.clone()))).collect::<BTreeMap<_,_>>(),"preflight":preflight,"execution":execution,"invocations":state.total()});log_event(summary.clone());summary
}
fn main(){std::fs::create_dir_all(out()).unwrap();let (_,o,p)=train("models-shared").unwrap();save("producer-summary.json",&serde_json::json!({"outcome_fingerprint":o.outcome_fingerprint,"package_fingerprint":p.package_fingerprint,"replayable_phases":o.replayable_phases}));}
