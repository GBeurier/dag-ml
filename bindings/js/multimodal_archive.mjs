/** Browser/Node consumer of an unchanged complete producer-owned Archive V2. */
import {strictJson} from "./n4m_multimodal_controller.mjs";

export async function replayMultimodalArchive({archiveBytes,readPortableArchiveV2,dagMl,controller,dataEnvelopes,request,outcomeId,runId}) {
  if(!(archiveBytes instanceof Uint8Array)||typeof readPortableArchiveV2!=="function")throw new Error("Actual Core archive reader and bytes are required");
  const archive=await readPortableArchiveV2(archiveBytes);
  const methods=archive.manifest.payloads?.methods;
  if(methods?.multimodal_pipelines?.length!==1||(methods.n4mm??[]).length||(methods.role_pipelines??[]).length)throw new Error("One complete U07 state archive is required");
  const packagePath=archive.manifest.replay.portable_predictor_package.member_path;
  const rawPackage=archive.members[packagePath];
  if(!(rawPackage instanceof Uint8Array))throw new Error("Core did not return exact portable package bytes");
  const packageText=new TextDecoder("utf-8",{fatal:true}).decode(rawPackage),packageValue=strictJson(packageText);
  // Native DAG validates all source/recipe/profile and exact member closure
  // before the numerical callback can hydrate any producer state.
  dagMl.validate_archive_v2_portable_payloads_json(JSON.stringify(archive.manifest),packageText,
    JSON.stringify(Object.fromEntries(Object.entries(archive.members).map(([name,bytes])=>[name,Array.from(bytes)]))));
  if(controller.allowFit!==false)throw new Error("Archive replay requires a target-free non-fitting controller");
  const manifest=controller.manifest();
  const replay=JSON.parse(dagMl.replay_training_package_json(packageText,JSON.stringify(request),JSON.stringify(dataEnvelopes),JSON.stringify([manifest]),outcomeId,runId,controller.callback));
  if(controller.models.size!==0)throw new Error("Native DAG failed to release the hydrated complete predictor");
  return {archiveId:archive.archiveId,archiveSha256:archive.archiveSha256,package:packageValue,replay,executionHost:controller.executionHost,signedController:controller.controllerId};
}
