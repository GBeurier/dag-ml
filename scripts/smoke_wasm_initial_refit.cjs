"use strict";

const fs = require("fs");
const path = require("path");

module.exports = function smokeInitialFullRefit(dagMl, repo) {
  const fixtureDir = path.join(repo, "crates", "dag-ml-core", "tests", "fixtures", "initial_full_refit");
  const fixtureJson = fs.readFileSync(path.join(fixtureDir, "package.json"), "utf8");
  const refitResultJson = fs.readFileSync(path.join(fixtureDir, "refit_node_result.json"), "utf8");
  dagMl.validate_initial_full_refit_package_json(fixtureJson);
  const fixture = JSON.parse(fixtureJson);
  const cohortRequest = {
    role: "external_test",
    relations: { records: [{
      observation_id: "obs.H001", sample_id: "sample:heldout:1",
      target_id: "target:heldout:1", group_id: "group:heldout",
      origin_sample_id: null, source_id: "nir", is_augmented: false,
    }] },
    target_names: ["y"], data_content_fingerprint: "a".repeat(64),
    target_content_fingerprint: "b".repeat(64),
  };
  const envelope = JSON.parse(dagMl.initial_full_refit_predict_envelope_json(
    fixtureJson, JSON.stringify(cohortRequest),
  ));
  if (envelope.schema_version !== 2 ||
      envelope.predict_cohort.physical_sample_ids[0] !== "sample:heldout:1") {
    throw new Error("WASM initial full-refit PREDICT envelope did not attest the heldout cohort");
  }

  const manifests = Object.values(fixture.effective_plan.controller_manifests);
  let calls = 0;
  const invoke = (_controllerId, taskJson, exactSeed) => {
    const task = JSON.parse(taskJson);
    const result = JSON.parse(refitResultJson);
    calls += 1;
    result.lineage.seed = null; // WASM bridge injects exact u64 from task.
    if (task.phase === "PREDICT") {
      result.artifacts = [];
      result.artifact_handles = {};
      result.lineage.artifact_refs = [];
      result.lineage.phase = "PREDICT";
      result.lineage.run_id = task.run_id;
      result.lineage.record_id = "lineage:wasm.initial.predict";
      result.predictions[0].prediction_id = "pred:wasm.initial.predict";
      result.predictions[0].sample_ids = ["sample:heldout:1"];
      result.predictions[0].values = [[7.0]];
    }
    if (typeof exactSeed !== "string" || !exactSeed.length) {
      throw new Error("WASM initial refit callback did not receive an exact seed");
    }
    return JSON.stringify(result);
  };
  const execution = JSON.parse(dagMl.execute_initial_full_refit_json(
    JSON.stringify(fixture.effective_plan), JSON.stringify(manifests),
    JSON.stringify(fixture.training_envelope), JSON.stringify(fixture.training_sample_ids),
    fixture.package_id, fixture.run_id, String(fixture.execution_root_seed), invoke,
  ));
  if (calls !== 1 || !execution.initial_full_refit_package.package_fingerprint) {
    throw new Error("WASM initial full-refit execution did not capture a package");
  }
  const capturedJson = JSON.stringify(execution.initial_full_refit_package);
  dagMl.validate_initial_full_refit_package_json(capturedJson);
  const replay = JSON.parse(dagMl.replay_initial_full_refit_json(
    capturedJson, JSON.stringify(envelope),
    JSON.stringify([execution.initial_full_refit_package.outputs[0].output_id]),
    JSON.stringify(execution.node_results[0].artifact_handles),
    "run:wasm.initial.predict", invoke,
  ));
  const prediction = replay.replay_outcome.outputs[0].prediction;
  if (calls !== 2 || prediction.sample_ids[0] !== "sample:heldout:1" || prediction.values[0][0] !== 7) {
    throw new Error("WASM initial full-refit replay did not route the named heldout prediction");
  }

  const tampered = JSON.parse(capturedJson);
  tampered.outputs[0].output_id = "output:forged";
  try {
    dagMl.replay_initial_full_refit_json(
      JSON.stringify(tampered), JSON.stringify(envelope),
      JSON.stringify([execution.initial_full_refit_package.outputs[0].output_id]),
      JSON.stringify(execution.node_results[0].artifact_handles),
      "run:wasm.initial.forged", invoke,
    );
    throw new Error("WASM initial full-refit replay accepted a forged package");
  } catch (error) {
    if (!String(error).includes("fingerprint")) throw error;
  }
  if (calls !== 2) throw new Error("WASM controller ran after package tamper");

  const graph = structuredClone(fixture.effective_plan.graph_plan.graph);
  const second = structuredClone(graph.nodes[0]);
  second.id = "model:second";
  graph.nodes.push(second);
  const campaign = structuredClone(fixture.effective_plan.campaign);
  const secondBinding = structuredClone(campaign.data_bindings["model:initial"][0]);
  secondBinding.node_id = "model:second";
  campaign.data_bindings["model:second"] = [secondBinding];
  const multiPlan = JSON.parse(dagMl.build_execution_plan_json(
    "plan:wasm.multi-output", JSON.stringify(graph), JSON.stringify(campaign),
    JSON.stringify(manifests),
  ));
  let multiCalls = 0;
  const multiInvoke = (_controllerId, taskJson) => {
    const task = JSON.parse(taskJson);
    const node = task.node_plan.node_id;
    const result = JSON.parse(refitResultJson);
    multiCalls++;
    result.lineage.seed = null;
    result.lineage.run_id = task.run_id;
    result.lineage.record_id = `lineage:wasm.multi.${task.phase}:${node}`;
    result.predictions[0].prediction_id = `pred:wasm.multi.${task.phase}:${node}`;
    if (node !== "model:initial") {
      const oldId = "artifact:model:initial:refit";
      const newId = `artifact:${node}:refit`;
      result.node_id = node;
      result.lineage.node_id = node;
      result.artifacts[0].id = newId;
      result.lineage.artifact_refs[0].id = newId;
      result.artifact_handles[newId] = result.artifact_handles[oldId];
      delete result.artifact_handles[oldId];
      result.predictions[0].producer_node = node;
    }
    if (task.phase === "PREDICT") {
      result.artifacts = [];
      result.artifact_handles = {};
      result.lineage.artifact_refs = [];
      result.lineage.phase = "PREDICT";
      result.lineage.run_id = task.run_id;
      result.lineage.record_id = `lineage:wasm.multi.predict:${node}`;
      result.predictions[0].prediction_id = `pred:wasm.multi.predict:${node}`;
      result.predictions[0].sample_ids = ["sample:heldout:1"];
      result.predictions[0].values = [[node === "model:initial" ? 7.0 : 8.0]];
    }
    return JSON.stringify(result);
  };
  const multiCapture = JSON.parse(dagMl.execute_initial_full_refit_json(
    JSON.stringify(multiPlan), JSON.stringify(manifests),
    JSON.stringify(fixture.training_envelope), JSON.stringify(fixture.training_sample_ids),
    "package:wasm.multi-output", "run:wasm.multi.refit", "12345", multiInvoke,
  ));
  const multiPackage = multiCapture.initial_full_refit_package;
  if (multiCalls !== 2 || multiPackage.outputs.length !== 2 || multiPackage.artifacts.length !== 2) {
    throw new Error("WASM initial refit failed to capture two independent outputs");
  }
  const multiEnvelope = dagMl.initial_full_refit_predict_envelope_json(
    JSON.stringify(multiPackage), JSON.stringify(cohortRequest),
  );
  const multiHandles = Object.assign({}, ...multiCapture.node_results.map(result => result.artifact_handles));
  const multiReplay = JSON.parse(dagMl.replay_initial_full_refit_json(
    JSON.stringify(multiPackage), multiEnvelope,
    JSON.stringify(multiPackage.outputs.map(output => output.output_id)),
    JSON.stringify(multiHandles), "run:wasm.multi.predict", multiInvoke,
  ));
  const multiOutputs = multiReplay.replay_outcome.outputs;
  if (multiCalls !== 4 || multiOutputs.length !== 2 ||
      multiOutputs[0].prediction.values[0][0] !== 7 ||
      multiOutputs[1].prediction.values[0][0] !== 8) {
    throw new Error("WASM initial refit failed to replay two named predictions");
  }

  const portableInvoke = (stats) => (_controllerId, taskJson, exactSeed) => {
    const task = JSON.parse(taskJson);
    if (task.operation) {
      if (task.schema_version !== 1) throw new Error("wrong portable artifact task version");
      if (task.operation === "export_artifact_payload") {
        stats.exports++;
        return JSON.stringify({operation:"exported_artifact_payload", schema_version:1, payload:[1,2,3]});
      }
      if (task.operation === "hydrate_artifact_payload") {
        if (JSON.stringify(task.payload) !== "[1,2,3]") throw new Error("wrong portable artifact bytes");
        stats.hydrates++;
        return JSON.stringify({operation:"hydrated_artifact_payload", schema_version:1,
          handle:{handle:442, kind:"model", owner_controller:"controller:model.mock"}});
      }
      if (task.operation === "release_hydrated_artifact_payload") {
        stats.releases++;
        return JSON.stringify({operation:"released_hydrated_artifact_payload", schema_version:1});
      }
      throw new Error(`unexpected portable operation ${task.operation}`);
    }
    stats.nodes++;
    const result = JSON.parse(multiInvoke(_controllerId, taskJson, exactSeed));
    if (task.phase === "REFIT") {
      result.artifacts[0].backend = "raw";
      result.artifacts[0].size_bytes = 3;
      result.artifacts[0].content_fingerprint = "039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81";
      result.lineage.artifact_refs[0] = structuredClone(result.artifacts[0]);
    }
    return JSON.stringify(result);
  };
  const captureStats = {nodes:0, exports:0, hydrates:0, releases:0};
  const rawCapture = JSON.parse(dagMl.execute_initial_full_refit_json(
    JSON.stringify(multiPlan), JSON.stringify(manifests),
    JSON.stringify(fixture.training_envelope), JSON.stringify(fixture.training_sample_ids),
    "package:wasm.multi-raw", "run:wasm.multi-raw.refit", "12345",
    portableInvoke(captureStats),
  ));
  const rawPackage = rawCapture.initial_full_refit_package;
  if (captureStats.nodes !== 2 || captureStats.exports !== 2 ||
      Object.keys(rawPackage.raw_artifact_payloads).length !== 2) {
    throw new Error("WASM failed to embed both raw model payloads");
  }
  const rawEnvelope = dagMl.initial_full_refit_predict_envelope_json(
    JSON.stringify(rawPackage), JSON.stringify(cohortRequest),
  );
  const replayStats = {nodes:0, exports:0, hydrates:0, releases:0};
  const rawReplay = JSON.parse(dagMl.replay_initial_full_refit_json(
    JSON.stringify(rawPackage), rawEnvelope,
    JSON.stringify(rawPackage.outputs.map(output => output.output_id)),
    "{}", "run:wasm.multi-raw.predict", portableInvoke(replayStats),
  ));
  if (rawReplay.replay_outcome.outputs.length !== 2 || replayStats.nodes !== 2 ||
      replayStats.hydrates !== 2 || replayStats.releases !== 2) {
    throw new Error("WASM failed to hydrate/release both raw models in a fresh host callback");
  }
};
