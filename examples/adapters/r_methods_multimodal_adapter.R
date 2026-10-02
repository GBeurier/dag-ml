#!/usr/bin/env Rscript
# Persistent raw ND controller. Numerical operations are exclusively in n4m.
suppressPackageStartupMessages(library(n4m))
require_ok <- function(ok, message) if (!isTRUE(ok)) stop(message, call.=FALSE)
order_sources <- c("nir", "image", "series", "metadata")
controller <- "controller:methods.r.multimodal"
plugin <- "dagml.methods.r.multimodal"
ids <- function(value) {
  value <- unlist(value, use.names=FALSE)
  require_ok(is.character(value) && length(value) > 0L && all(nzchar(value)) && !anyDuplicated(value), "Unique nonempty IDs required")
  value
}
closed_keys <- function(value, expected) is.list(value) && !is.null(names(value)) && identical(sort(names(value)), sort(expected))
same <- function(a, b) {
  if (is.list(a) && is.list(b)) {
    if (is.null(names(a)) != is.null(names(b)) || length(a) != length(b)) return(FALSE)
    if (!is.null(names(a))) {
      if (!identical(sort(names(a)), sort(names(b)))) return(FALSE)
      return(all(vapply(names(a), function(key) same(a[[key]], b[[key]]), logical(1))))
    }
    return(all(vapply(seq_along(a), function(i) same(a[[i]], b[[i]]), logical(1))))
  }
  identical(a,b) || (is.numeric(a) && is.numeric(b) && identical(as.double(a),as.double(b)))
}
encode <- function(value) jsonlite::toJSON(value, auto_unbox=TRUE, null="null", digits=I(17L), force=TRUE)
# A bounded lexical walk catches duplicate fields before jsonlite drops them.
json_tokens <- function(text) {
  require_ok(nchar(text,type="bytes") <= 134217728L, "JSON transport budget exceeded")
  i <- 1L; n <- nchar(text); members <- list()
  skip <- function() { while (i <= n && grepl("[[:space:]]",substr(text,i,i))) i <<- i+1L }
  string <- function() {
    first <- i; i <<- i+1L
    while (i <= n) { ch <- substr(text,i,i); i <<- i+1L; if (ch == "\\") i <<- i+1L else if (ch == '"') return(jsonlite::fromJSON(substr(text,first,i-1L))) }
    stop("Unterminated JSON string",call.=FALSE)
  }
  value <- function(depth=0L) {
    require_ok(depth <= 128L,"JSON nesting budget exceeded"); skip(); ch <- substr(text,i,i)
    if(ch == '"') { string(); return(invisible(NULL)) }
    if(ch %in% c("{","[")) {
      obj <- ch == "{"; end <- if(obj) "}" else "]"; i <<- i+1L; seen <- character(); skip()
      if(substr(text,i,i)==end) {i <<- i+1L;return(invisible(NULL))}
      repeat {
        key <- NULL
        if(obj) { require_ok(substr(text,i,i)=='"',"Expected JSON key");key <- string();require_ok(!key %in% seen,"Duplicate JSON field");seen <- c(seen,key);skip();require_ok(substr(text,i,i)==":","Expected colon");i <<- i+1L;skip() }
        first <- i; value(depth+1L); last <- i-1L
        if(obj && depth==0L) members[[key]] <<- substr(text,first,last)
        skip();if(substr(text,i,i)==end){i <<- i+1L;break};require_ok(substr(text,i,i)==",","Expected comma");i <<- i+1L;skip()
      }
      return(invisible(NULL))
    }
    token <- regmatches(substr(text,i,n),regexpr("^(null|true|false|-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?)",substr(text,i,n),perl=TRUE))
    require_ok(length(token)==1L&&nzchar(token),"Invalid JSON token")
    if(!token %in% c("null","true","false")) require_ok(is.finite(as.double(token)),"Nonfinite JSON number")
    i <<- i+nchar(token)
  }
  value();skip();require_ok(i==n+1L,"Trailing JSON data");members
}
decode <- function(text) {json_tokens(text);jsonlite::fromJSON(text,simplifyVector=FALSE)}
number <- function(value, lower=0, upper=Inf, integer=FALSE) is.numeric(value)&&length(value)==1L&&is.finite(value)&&value>=lower&&value<=upper&&(!integer||value==floor(value))
validate_recipe <- function(recipe, schemas) {
  require_ok(closed_keys(recipe,c("schema_version","fusion","source_order","encoders","source_weights","model"))&&number(recipe$schema_version,1,1,TRUE)&&identical(recipe$fusion,"early")&&identical(ids(recipe$source_order),order_sources),"Closed early fusion recipe required")
  require_ok(closed_keys(schemas,order_sources)&&closed_keys(recipe$encoders,order_sources)&&closed_keys(recipe$source_weights,order_sources),"Complete four source declarations required")
  repr <- c("signal_1d","rgb_image","series_mv","tabular_mixed")
  for(i in seq_along(order_sources)) {
    name <- order_sources[[i]]; schema <- schemas[[name]];shape <- unlist(schema$input_shape,use.names=FALSE)
    require_ok(closed_keys(schema,c("representation_id","input_shape","dtype","identity"))&&identical(schema$representation_id,repr[[i]])&&is.numeric(shape)&&length(shape)>0L&&length(shape)<=7L&&all(is.finite(shape)&shape>0&shape==floor(shape))&&prod(shape)<=1048576&&(name!="metadata"||identical(as.double(shape),2)),"Fixed raw source schema required")
    require_ok(is.character(schema$dtype)&&length(schema$dtype)==1L&&nchar(schema$dtype,type="bytes")%in%1:128&&is.character(schema$identity)&&length(schema$identity)==1L&&nchar(schema$identity,type="bytes")>0L&&nchar(schema$identity,type="bytes")<=1048576,"Source identity budget exceeded");decode(schema$identity)
    require_ok(number(recipe$source_weights[[name]]),"Finite nonnegative source weight required")
  }
  require_ok(same(recipe$encoders$nir,list(kind="standard_scaler",with_mean=TRUE,with_std=TRUE))&&same(recipe$encoders$metadata,list(kind="column_transformer",numeric_columns=list(0L),categorical_columns=list(1L),with_mean=TRUE,with_std=TRUE,handle_unknown="ignore",sparse_output=FALSE,drop=NULL)),"Closed scaler/mixed encoder required")
  for(name in c("image","series")) {e <- recipe$encoders[[name]];require_ok(closed_keys(e,c("kind","n_components","whiten","random_state"))&&identical(e$kind,"tensor_pca")&&number(e$n_components,1,min(2147483647,prod(unlist(schemas[[name]]$input_shape))),TRUE)&&identical(e$whiten,FALSE)&&number(e$random_state,0,4294967295,TRUE),"Declared unwhitened PCA required")}
  m <- recipe$model;require_ok(closed_keys(m,c("method_id","params"))&&identical(m$method_id,"models.regularized.ridge")&&closed_keys(m$params,c("alpha","center_x","center_y","scale_x"))&&number(m$params$alpha)&&identical(m$params$center_x,TRUE)&&identical(m$params$center_y,TRUE)&&identical(m$params$scale_x,FALSE),"Closed native Ridge recipe required")
}
recipe_for <- function(operator, params=list()) {
  if(is.null(params))params <- list()
  require_ok(closed_keys(operator,c("type","recipe","source_schemas"))&&identical(operator$type,"N4mMultimodalPipeline")&&is.list(params)&&all(names(params)%in%c("model__alpha","source_weights__image","transformers__image__n_components")),"Explicit operator/effective parameters required")
  recipe <- operator$recipe
  if(!is.null(params$model__alpha))recipe$model$params$alpha <- params$model__alpha
  if(!is.null(params$source_weights__image))recipe$source_weights$image <- params$source_weights__image
  if(!is.null(params$transformers__image__n_components))recipe$encoders$image$n_components <- params$transformers__image__n_components
  validate_recipe(recipe,operator$source_schemas);recipe
}
byte_array <- function(value,maximum=134217728L) {
  value <- unlist(value,use.names=FALSE);require_ok(is.numeric(value)&&length(value)>0L&&length(value)<=maximum&&all(is.finite(value)&value>=0&value<=255&value==floor(value)),"Exact bounded byte array required");as.raw(value)
}
validate_wrapper <- function(saved) {
  require_ok(closed_keys(saved,c("schema","node_id","params_fingerprint","target_names","recipe","source_schemas","state"))&&identical(saved$schema,"dagml.methods.multimodal.v1")&&length(ids(saved$target_names))==1L,"Closed complete predictor wrapper required")
  state <- byte_array(saved$state,67108864L);require_ok(length(state)>=28L&&identical(as.integer(state[1:12]),c(78L,52L,77L,70L,1L,0L,0L,0L,2L,0L,0L,0L)),"Unsupported N4MF state header");validate_recipe(saved$recipe,saved$source_schemas);state
}
main <- function() {
  config <- decode(readChar(Sys.getenv("DAGML_METHODS_MULTIMODAL_CONFIG"),file.info(Sys.getenv("DAGML_METHODS_MULTIMODAL_CONFIG"))$size,useBytes=TRUE))
  if(!is.null(config$controller_id))controller <<- config$controller_id
  owners <- setNames(c("python","wasm","r","octave"),paste0("controller:methods.",c("python","wasm","r","octave"),".multimodal"))
  require_ok(controller%in%names(owners)&&(!isTRUE(config$allow_fit)||identical(owners[[controller]],"r")),"Closed producer owner required; fitting requires R ownership")
  plugin <<- paste0("dagml.methods.",owners[[controller]],".multimodal")
  require_ok(same(config$manifest,config$trusted_manifest)&&identical(config$manifest$controller_id,controller)&&identical(config$manifest$controller_version,"1.0.0")&&identical(config$manifest$operator_kind,"model"),"Current manifest differs from independently trusted controller")
  require_ok(is.logical(config$allow_fit)&&length(config$allow_fit)==1L&&config$allow_fit==!is.null(config$targets),"Replay cannot receive targets")
  target_names <- ids(config$target_names);source_ids <- ids(config$source_ids);require_ok(length(target_names)==1L&&length(source_ids)==4L&&closed_keys(config$sources,order_sources),"One target/four raw sources required")
  sources <- config$sources
  for(name in order_sources) {
    s <- sources[[name]];s$sample_ids <- ids(s$sample_ids);shape <- as.integer(unlist(s$descriptor$input_shape));n <- length(s$sample_ids)
    if(name=="metadata") {
      require_ok(is.list(s$rows)&&length(s$rows)==n&&all(vapply(s$rows,function(row)is.list(row)&&length(row)==2L&&is.character(row[[2L]])&&nchar(row[[2L]],type="bytes")<=1048576,logical(1))),"Raw UTF-8 metadata rows required")
      s$values <- do.call(rbind,lapply(s$rows,function(row)c(as.character(row[[1L]]),row[[2L]])))
    } else {
      values <- unlist(s$data,use.names=FALSE);require_ok(number(length(values),1,16777216,TRUE)&&identical(as.integer(unlist(s$shape)),c(n,shape))&&length(values)==prod(c(n,shape))&&all(is.finite(values)),"Raw tensor shape/budget mismatch")
      # Transport is row-major; conversion only changes storage strides.
      s$values <- aperm(array(as.double(values),dim=rev(c(n,shape))),rev(seq_along(c(n,shape))))
    }
    sources[[name]] <- s
  }
  for(node in names(config$operators)){op <- config$operators[[node]];recipe_for(op,config$node_params[[node]]);require_ok(same(op$source_schemas,lapply(sources,function(s)s$descriptor)),"Current independent raw schema mismatch")}
  models <- new.env(parent=emptyenv());artifacts <- new.env(parent=emptyenv());next_handle <- 0L;closed <- FALSE
  audit <- function(operation,...) {if(!is.null(config$audit_path))cat(encode(c(list(operation=operation),list(...))),"\n",file=config$audit_path,append=TRUE,sep="")}
  dispose <- function(entry){n4m::n4m_close(entry$model);audit("dispose")}
  close_all <- function(){if(closed)return(invisible(NULL));failure <- NULL;for(key in ls(models,all.names=TRUE)){entry <- models[[key]];rm(list=key,envir=models);tryCatch(dispose(entry),error=function(e){if(is.null(failure))failure <<- e})};rm(list=ls(artifacts,all.names=TRUE),envir=artifacts);closed <<- TRUE;if(!is.null(failure))stop(failure)}
  on.exit(close_all(),add=TRUE)
  keep <- function(entry){require_ok(next_handle<2147483647L,"Handle space exhausted");next_handle <<- next_handle+1L;models[[as.character(next_handle)]] <- entry;list(handle=next_handle,kind="model",owner_controller=controller)}
  handle_key <- function(handle){require_ok(identical(handle$kind,"model")&&identical(handle$owner_controller,controller)&&number(handle$handle,1,2147483647,TRUE),"Foreign model handle");key <- as.character(handle$handle);require_ok(exists(key,models,inherits=FALSE),"Unknown model handle");key}
  features <- function(task,partition){views <- Filter(function(view)identical(view$partition,partition),task$data_views);require_ok(length(views)==1L,"One native raw view required");view <- views[[1L]];samples <- ids(view$sample_ids);require_ok(identical(ids(view$source_ids),source_ids)&&!isTRUE(view$include_augmented)&&(partition%in%c("fold_validation","predict")||!isTRUE(view$include_excluded))&&length(view$columns)==0L,"Raw native source view mismatch");blocks <- list();for(name in order_sources){s <- sources[[name]];rows <- match(samples,s$sample_ids);require_ok(!anyNA(rows),"Unknown raw sample ID");if(name=="metadata")blocks[[name]] <- s$values[rows,,drop=FALSE] else blocks[[name]] <- do.call(`[`,c(list(s$values,rows),rep(list(TRUE),length(dim(s$values))-1L),list(drop=FALSE)))};list(sample_ids=samples,blocks=blocks)}
  targets <- function(samples){require_ok(!is.null(config$targets),"Replay cannot read targets");rows <- match(samples,ids(config$targets$sample_ids));require_ok(!anyNA(rows),"Unknown target sample ID");values <- vapply(config$targets$values,function(row)as.double(unlist(row)),double(1));matrix(values[rows],ncol=1L)}
  result <- function(task,block,model,refs=list()) {
    values <- as.double(stats::predict(model,block$blocks,source_schemas=config$operators[[task$node_plan$node_id]]$source_schemas));require_ok(length(values)==length(block$sample_ids)&&all(is.finite(values)),"Native prediction shape mismatch")
    node <- task$node_plan;fallback <- function(x,y)if(is.null(x))y else x
    out <- list(node_id=node$node_id,outputs=setNames(list(),character()),artifacts=refs,artifact_handles=setNames(list(),character()),predictions=list(list(producer_node=node$node_id,partition=if(task$phase=="FIT_CV")"validation" else "final",fold_id=task$fold_id,sample_ids=as.list(block$sample_ids),values=lapply(values,function(v)list(v)),target_names=as.list(target_names))),lineage=list(record_id=paste("lineage:methods-multimodal",task$run_id,node$node_id,task$phase,fallback(task$variant_id,"base"),fallback(task$fold_id,"full"),sep=":"),run_id=task$run_id,node_id=node$node_id,phase=task$phase,controller_id=controller,controller_version="1.0.0",variant_id=task$variant_id,fold_id=task$fold_id,branch_path=task$branch_path,input_lineage=list(),artifact_refs=refs,params_fingerprint=node$params_fingerprint,data_model_shape_fingerprint=NULL,aggregation_policy_fingerprint=NULL,seed="__DAGML_U64_SEED__",unsafe_flags=list(),metrics=setNames(list(),character()),loss_attestations=list(),early_stopping_records=list()))
    if(!is.null(config$targets))out$regression_targets <- list(list(level="sample",unit_ids=lapply(block$sample_ids,function(id)list(level="sample",id=id)),values=lapply(as.double(targets(block$sample_ids)),function(v)list(v)),target_names=as.list(target_names)))
    if(task$phase%in%c("FIT_CV","REFIT")&&any(vapply(task$data_views,function(view)identical(view$partition,"predict"),logical(1)))) {
      test <- features(task,"predict");predicted <- as.double(stats::predict(model,test$blocks,source_schemas=config$operators[[node$node_id]]$source_schemas));require_ok(length(predicted)==length(test$sample_ids)&&all(is.finite(predicted)),"Native test prediction width mismatch")
      out$predictions <- c(out$predictions,list(list(producer_node=node$node_id,partition="test",fold_id=task$fold_id,sample_ids=as.list(test$sample_ids),values=lapply(predicted,function(value)list(value)),target_names=as.list(target_names))))
      if(!is.null(config$targets))out$regression_targets <- c(out$regression_targets,list(list(level="sample",unit_ids=lapply(test$sample_ids,function(id)list(level="sample",id=id)),values=lapply(as.double(targets(test$sample_ids)),function(value)list(value)),target_names=as.list(target_names))))
    }
    audit(task$phase,node_id=node$node_id,sample_ids=as.list(block$sample_ids));out
  }
  portable <- function(task) {
    require_ok(number(task$schema_version,1,1,TRUE),"Unsupported artifact bridge")
    if(task$operation=="export_artifact_payload"){require_ok(exists(task$artifact_id,artifacts,inherits=FALSE),"Unknown artifact");audit("export");return(list(operation="exported_artifact_payload",schema_version=1L,payload=as.list(as.integer(artifacts[[task$artifact_id]]))))}
    if(task$operation=="release_hydrated_artifact_payload"){key <- handle_key(task$handle);entry <- models[[key]];rm(list=key,envir=models);dispose(entry);audit("release");return(list(operation="released_hydrated_artifact_payload",schema_version=1L))}
    require_ok(task$operation=="hydrate_artifact_payload","Unknown artifact operation");payload <- byte_array(task$payload);request <- task$request;ref <- request$artifact;sha <- digest::digest(payload,algo="sha256",serialize=FALSE)
    require_ok(identical(request$controller_id,controller)&&identical(ref$controller_id,controller)&&identical(ref$kind,"methods_multimodal_pipeline")&&identical(ref$backend,"raw")&&identical(ref$plugin,plugin)&&identical(ref$plugin_version,"1.0.0")&&is.null(ref$native_predictor_descriptor)&&is.null(ref$native_estimator_descriptor)&&identical(ref$content_fingerprint,sha)&&identical(ref$uri,paste0("artifacts/",sha,".json"))&&number(ref$size_bytes,length(payload),length(payload),TRUE),"RAW owner/plugin/hash/URI/size mismatch")
    saved <- decode(rawToChar(payload));state <- validate_wrapper(saved);operator <- config$operators[[saved$node_id]]
    require_ok(!is.null(operator)&&identical(saved$node_id,request$node_id)&&identical(saved$params_fingerprint,request$params_fingerprint)&&identical(ids(saved$target_names),target_names)&&same(saved$source_schemas,operator$source_schemas)&&same(saved$recipe,recipe_for(operator,config$node_params[[saved$node_id]])),"Selected recipe/schema/node mismatch before hydration")
    model <- n4m::n4m_multimodal_pipeline_from_state(state,saved$recipe,saved$source_schemas);retained <- FALSE;on.exit(if(!retained)n4m::n4m_close(model),add=TRUE);handle <- keep(list(model=model,saved=saved,artifact=ref));retained <- TRUE;audit("hydrate");list(operation="hydrated_artifact_payload",schema_version=1L,handle=handle)
  }
  invoke <- function(task) {
    node <- task$node_plan;operator <- config$operators[[node$node_id]];require_ok(identical(node$kind,"model")&&identical(node$controller_id,controller)&&identical(node$controller_version,"1.0.0")&&!is.null(operator)&&task$phase%in%c("FIT_CV","REFIT","PREDICT"),"Foreign native node/phase")
    require_ok(length(task$prediction_inputs)==0L&&length(task$data_view_receipts)==0L&&length(task$required_loss_attestations)==0L&&is.null(task$residual_targets)&&(is.null(task$fit_influence)||(identical(task$fit_influence$mechanism,"uniform_rows")&&length(task$fit_influence$row_weights)==0L)),"Generated/OOF/loss/residual/nonuniform inputs unsupported")
    recipe <- recipe_for(operator,node$params)
    if(task$phase=="PREDICT"){require_ok(length(task$artifact_inputs)==1L,"One complete predictor required");key <- names(task$artifact_inputs)[[1L]];entry <- models[[handle_key(task$input_handles[[key]])]];input <- task$artifact_inputs[[key]];require_ok(identical(input$node_id,node$node_id)&&identical(input$controller_id,controller)&&identical(input$params_fingerprint,node$params_fingerprint)&&same(input$artifact,entry$artifact)&&identical(entry$saved$node_id,node$node_id)&&identical(entry$saved$params_fingerprint,node$params_fingerprint)&&same(entry$saved$recipe,recipe)&&same(entry$saved$source_schemas,operator$source_schemas),"PREDICT binding mismatch");return(result(task,features(task,"predict"),entry$model))}
    require_ok(config$allow_fit,"Fitting disabled for replay");train <- features(task,if(task$phase=="FIT_CV")"fold_train" else "full_train");valid <- if(task$phase=="FIT_CV")features(task,"fold_validation") else train;require_ok(task$phase!="FIT_CV"||length(intersect(train$sample_ids,valid$sample_ids))==0L,"Training validation overlap")
    model <- n4m::n4m_multimodal_pipeline(recipe,operator$source_schemas);retained <- FALSE;on.exit(if(!retained){n4m::n4m_close(model);audit("dispose")},add=TRUE);n4m::n4m_fit(model,train$blocks,targets(train$sample_ids));audit("fit",node_id=node$node_id,fold_id=task$fold_id,sample_ids=as.list(train$sample_ids))
    if(task$phase=="FIT_CV")return(result(task,valid,model))
    saved <- list(schema="dagml.methods.multimodal.v1",node_id=node$node_id,params_fingerprint=node$params_fingerprint,target_names=as.list(target_names),recipe=recipe,source_schemas=operator$source_schemas,state=as.list(as.integer(n4m::n4m_export_state(model))));validate_wrapper(saved);payload <- charToRaw(enc2utf8(encode(saved)));require_ok(length(payload)<=134217728L,"Complete payload budget exceeded");sha <- digest::digest(payload,algo="sha256",serialize=FALSE);id <- paste("artifact:methods.multimodal",task$run_id,node$node_id,if(is.null(task$variant_id))"base" else task$variant_id,"refit",sep=":");require_ok(!exists(id,artifacts,inherits=FALSE),"Duplicate REFIT artifact");ref <- list(id=id,kind="methods_multimodal_pipeline",controller_id=controller,backend="raw",uri=paste0("artifacts/",sha,".json"),content_fingerprint=sha,size_bytes=length(payload),plugin=plugin,plugin_version="1.0.0");out <- result(task,valid,model,list(ref));handle <- keep(list(model=model,saved=saved,artifact=ref));artifacts[[id]] <- payload;out$artifact_handles[[id]] <- handle;retained <- TRUE;out
  }
  input <- file("stdin",open="r");on.exit(close(input),add=TRUE)
  repeat {
    line <- readLines(input,n=1L,warn=FALSE);if(!length(line))break
    reply <- tryCatch({frame <- decode(line);require_ok(number(frame$schema_version,1,1,TRUE),"Unsupported process schema");require_ok(!closed,"Controller closed");
      if(frame$type=="init"){require_ok(identical(frame$controller_id,controller),"Foreign controller init");list(type="ack",schema_version=1L,status="initialized",runtime=list(execution_host="r",signed_controller=controller,r=R.version.string,n4m=find.package("n4m"),dlls=lapply(getLoadedDLLs(),function(dll)dll[["path"]])))}
      else if(frame$type=="task"){seed <- json_tokens(json_tokens(line)$task)$seed;require_ok(is.character(seed)&&length(seed)==1L&&(identical(seed,"null")||(grepl("^(0|[1-9][0-9]*)$",seed)&&(nchar(seed)<20L||(nchar(seed)==20L&&seed<="18446744073709551615")))),"Exact null or u64 seed token required");out <- encode(list(type="result",schema_version=1L,result=invoke(frame$task)));cat(sub('"seed":"__DAGML_U64_SEED__"',paste0('"seed":',seed),out,fixed=TRUE),"\n",sep="");flush(stdout());NULL}
      else if(frame$type=="portable_artifact")list(type="portable_artifact",schema_version=1L,result=portable(frame$task))
      else if(frame$type=="close"){close_all();list(type="ack",schema_version=1L,status="closed")}
      else stop("Unsupported process frame",call.=FALSE)
    },error=function(e)list(type="error",schema_version=1L,error=list(code="r_methods_multimodal_refusal",message=conditionMessage(e))))
    if(!is.null(reply)){cat(encode(reply),"\n",sep="");flush(stdout())};if(closed)break
  }
}
main()
