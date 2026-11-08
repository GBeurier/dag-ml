/** Core-validated native Package2 replay in a Core V1 container. */
import {strictJson} from "./n4m_multimodal_controller.mjs";

// Core's reader validates the selected native scientific closure through Rust
// DAG before returning exact package bytes. No old physical manifest is rebuilt.
export function multimodalArchivePackage(archive) {
  const rawPackage=archive.package;
  if(!(rawPackage instanceof Uint8Array))throw new Error("Core did not return exact validated native package bytes");
  const packageText=new TextDecoder("utf-8",{fatal:true}).decode(rawPackage),packageValue=strictJson(packageText);
  const artifacts=packageValue.execution_bundle?.refit_artifacts;
  if(packageValue.schema_version!==2||artifacts?.length!==1||artifacts[0].artifact.kind!=="methods_multimodal_pipeline")throw new Error("One complete U07 state package is required");
  return {packageText,packageValue};
}

export async function replayMultimodalArchive({archiveBytes,readPortableArchiveV2,dagMl,controller,dataEnvelopes,request,outcomeId,runId,entrypoint}) {
  if(!(archiveBytes instanceof Uint8Array)||typeof readPortableArchiveV2!=="function")throw new Error("Actual Core archive reader and bytes are required");
  const archive=await readPortableArchiveV2(archiveBytes,{entrypoint});
  const {packageText,packageValue}=multimodalArchivePackage(archive);
  if(controller.allowFit!==false)throw new Error("Archive replay requires a target-free non-fitting controller");
  const manifest=controller.manifest();
  const replay=JSON.parse(dagMl.replay_training_package_json(packageText,JSON.stringify(request),JSON.stringify(dataEnvelopes),JSON.stringify([manifest]),outcomeId,runId,controller.callback));
  if(controller.models.size!==0)throw new Error("Native DAG failed to release the hydrated complete predictor");
  return {archiveId:archive.archiveId,archiveSha256:archive.archiveSha256,package:packageValue,replay,executionHost:controller.executionHost,signedController:controller.controllerId};
}
