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
};
