#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");

const repo = path.resolve(__dirname, "..");
const pkgDir = path.resolve(process.argv[2] || path.join(repo, "target", "wasm", "dag-ml-wasm"));
const dagMl = require(path.join(pkgDir, "dag_ml_wasm.js"));
const SHARED_FOLD_SET_FINGERPRINT =
  "54d3185d6c628ef0df848828a8d8ae650222a283a78bbd3ab3bc2256f222c05c";
const REQUIRED_DTS_EXPORTS = [
  "build_execution_plan_json",
  "build_execution_plan_with_training_losses_json",
  "compile_pipeline_dsl_artifact_json",
  "compile_pipeline_dsl_graph_json",
  "contract_manifest_json",
  "dag_ml_version",
  "derive_controller_manifest_json",
  "derive_controller_manifest_list_json",
  "execute_execution_plan_phase_json",
  "fold_set_fingerprint_json",
  "loss_execution_attestation_json",
  "validate_fold_set_json",
];

function assertPackageMetadata(expectedVersion) {
  const packageJson = JSON.parse(fs.readFileSync(path.join(pkgDir, "package.json"), "utf8"));
  if (packageJson.name !== "dag-ml-wasm") {
    throw new Error("WASM package.json has wrong package name");
  }
  if (packageJson.version !== expectedVersion) {
    throw new Error("WASM package.json version does not match contract manifest");
  }
  if (packageJson.main !== "dag_ml_wasm.js" && packageJson.module !== "dag_ml_wasm.js") {
    throw new Error("WASM package.json does not point to dag_ml_wasm.js");
  }
  if (packageJson.types !== "dag_ml_wasm.d.ts") {
    throw new Error("WASM package.json does not point to dag_ml_wasm.d.ts");
  }
  for (const filename of ["dag_ml_wasm.js", "dag_ml_wasm_bg.wasm", "dag_ml_wasm.d.ts"]) {
    if (!fs.existsSync(path.join(pkgDir, filename))) {
      throw new Error(`WASM package is missing ${filename}`);
    }
  }
  const dts = fs.readFileSync(path.join(pkgDir, "dag_ml_wasm.d.ts"), "utf8");
  for (const exportName of REQUIRED_DTS_EXPORTS) {
    if (!dts.includes(`export function ${exportName}(`)) {
      throw new Error(`WASM TypeScript declarations are missing ${exportName}()`);
    }
  }
  if (!dts.includes("export class LocalImplementationRegistry")) {
    throw new Error("WASM TypeScript declarations are missing LocalImplementationRegistry");
  }
  if (!dts.includes("bind_training_loss(")) {
    throw new Error("WASM TypeScript declarations are missing task-level loss binding");
  }
  if (!dts.includes("export class TrainingLossBinding")) {
    throw new Error("WASM TypeScript declarations are missing TrainingLossBinding");
  }
  if (!dts.includes("bind_training_loss(node_task_json: string, role_index: number): TrainingLossBinding")) {
    throw new Error("WASM task-level loss binding has a weak TypeScript return type");
  }
}

function parseErrorDescriptor(error) {
  const payload = typeof error === "string" ? error : String(error);
  const start = payload.indexOf("{");
  if (start < 0) {
    throw new Error(`error payload is not JSON: ${payload}`);
  }
  return JSON.parse(payload.slice(start));
}

const dslJson = fs.readFileSync(
  path.join(repo, "examples", "pipeline_dsl_generation.json"),
  "utf8",
);
const sharedFoldSetJson = fs.readFileSync(
  path.join(repo, "examples", "fixtures", "shared", "fold_set_cv_partition.json"),
  "utf8",
);
const localImplementations = JSON.parse(
  fs.readFileSync(
    path.join(
      repo,
      "examples",
      "fixtures",
      "criteria",
      "javascript_local_implementations.v1.json",
    ),
    "utf8",
  ),
);
dagMl.validate_pipeline_dsl_json(dslJson);
const manifest = JSON.parse(dagMl.contract_manifest_json());
if (manifest.crate !== "dag-ml") {
  throw new Error("contract manifest has wrong crate name");
}
if (manifest.version !== dagMl.dag_ml_version()) {
  throw new Error("contract manifest version does not match WASM version");
}
assertPackageMetadata(manifest.version);
if (!manifest.wasm_exports.includes("compile_pipeline_dsl_artifact_json")) {
  throw new Error("contract manifest is missing WASM DSL export");
}
if (!manifest.wasm_exports.includes("derive_controller_manifest_json")) {
  throw new Error("contract manifest is missing WASM controller-derivation export");
}
if (!manifest.capabilities.includes("structured_error_descriptors")) {
  throw new Error("contract manifest is missing structured error capability");
}
if (!manifest.capabilities.includes("process_local_implementation_registry")) {
  throw new Error("contract manifest is missing local implementation capability");
}
if (!manifest.capabilities.includes("bind_training_losses_to_execution_plan")) {
  throw new Error("contract manifest is missing native training-loss plan binding");
}
if (!manifest.capabilities.includes("execute_execution_plan_phase")) {
  throw new Error("contract manifest is missing execution-plan runtime capability");
}
if (manifest.shared.fold_set_fixture_fingerprint !== SHARED_FOLD_SET_FINGERPRINT) {
  throw new Error("contract manifest shared fold fingerprint drifted");
}

const implementations = new dagMl.LocalImplementationRegistry();
const lossReferenceJson = JSON.stringify(localImplementations.loss_reference);
const trainingLossRoleJson = JSON.stringify(localImplementations.training_loss_role);
const metricReferenceJson = JSON.stringify(localImplementations.metric_reference);
const lossCalls = [];
const asymmetricLoss = (target, prediction) => {
  lossCalls.push([target, prediction]);
  return prediction >= target ? prediction - target : 2 * (target - prediction);
};
implementations.register_loss(lossReferenceJson, asymmetricLoss);
try {
  implementations.register_loss(
    JSON.stringify(localImplementations.foreign_loss_reference),
    asymmetricLoss,
  );
  throw new Error("JavaScript registry accepted a Python binding descriptor");
} catch (error) {
  if (!String(error).includes("binding:javascript")) {
    throw error;
  }
}
for (const phase of ["FIT_CV", "REFIT"]) {
  const requiredAttestation = JSON.parse(
    dagMl.loss_execution_attestation_json(trainingLossRoleJson, phase),
  );
  const task = {
    run_id: `run:javascript-local-${phase.toLowerCase()}`,
    node_plan: {
      node_id: "model:custom",
      kind: "model",
      controller_id: "controller:javascript-local",
      controller_version: "1.0.0",
      supported_phases: ["FIT_CV", "REFIT"],
      controller_capabilities: [
        "deterministic",
        "supports_configurable_loss",
        "supports_custom_loss",
        "supports_differentiable_loss",
      ],
      training_losses: [localImplementations.training_loss_role],
      fit_scope: "fold_train",
      rng_policy: "uses_core_seed",
      artifact_policy: "serializable",
      input_nodes: [],
      output_nodes: [],
      shape_plan: null,
      data_bindings: [],
      params: {},
      params_fingerprint: "0".repeat(64),
    },
    phase,
    variant_id: null,
    variant: null,
    fold_id: phase === "FIT_CV" ? "fold:0" : null,
    branch_path: [],
    input_handles: {},
    data_views: {},
    prediction_inputs: {},
    artifact_inputs: {},
    required_loss_attestations: [requiredAttestation],
    seed: 42,
  };
  const binding = implementations.bind_training_loss(
    JSON.stringify(task),
    0,
  );
  const resolved = binding.invoke;
  if (resolved(4, 1.5) !== 5) {
    throw new Error(`JavaScript custom loss returned the wrong value during ${phase}`);
  }
  const attestation = JSON.parse(binding.required_attestation_json);
  if (
    attestation.phase !== phase ||
    attestation.attestation_fingerprint !== requiredAttestation.attestation_fingerprint
  ) {
    throw new Error(`JavaScript custom loss attestation drifted during ${phase}`);
  }
  binding.free();
  const tamperedTask = structuredClone(task);
  tamperedTask.required_loss_attestations[0].loss_id = "example.loss.tampered@1";
  try {
    implementations.bind_training_loss(JSON.stringify(tamperedTask), 0);
    throw new Error("JavaScript registry bound a tampered NodeTask loss requirement");
  } catch (error) {
    if (!String(error).includes("do not match")) {
      throw error;
    }
  }
}
if (lossCalls.length !== 2) {
  throw new Error("JavaScript custom loss was not invoked for FIT_CV and REFIT");
}
try {
  const predictTask = {
    run_id: "run:javascript-local-predict",
    node_plan: {
      node_id: "model:custom",
      kind: "model",
      controller_id: "controller:javascript-local",
      controller_version: "1.0.0",
      supported_phases: ["FIT_CV", "REFIT", "PREDICT"],
      controller_capabilities: ["supports_configurable_loss", "supports_custom_loss"],
      training_losses: [localImplementations.training_loss_role],
      fit_scope: "fold_train",
      rng_policy: "uses_core_seed",
      artifact_policy: "serializable",
      input_nodes: [],
      output_nodes: [],
      shape_plan: null,
      data_bindings: [],
      params: {},
      params_fingerprint: "0".repeat(64),
    },
    phase: "PREDICT",
    variant_id: null,
    variant: null,
    fold_id: null,
    branch_path: [],
    input_handles: {},
    data_views: {},
    prediction_inputs: {},
    artifact_inputs: {},
    required_loss_attestations: [],
    seed: 42,
  };
  implementations.bind_training_loss(JSON.stringify(predictTask), 0);
  throw new Error("JavaScript custom loss bound during PREDICT");
} catch (error) {
  if (!String(error).includes("FIT_CV or REFIT")) {
    throw error;
  }
}
for (const invalidRoleIndex of [
  -1,
  0.5,
  Number.NaN,
  Number.POSITIVE_INFINITY,
  2 ** 53,
  null,
  false,
  "0",
  undefined,
  {},
]) {
  try {
    implementations.bind_training_loss("{}", invalidRoleIndex);
    throw new Error(`JavaScript registry accepted invalid role index ${invalidRoleIndex}`);
  } catch (error) {
    if (!String(error).includes("non-negative safe integer")) {
      throw error;
    }
  }
}

const biasMetric = (targets, predictions) =>
  predictions.reduce((sum, prediction, index) => sum + prediction - targets[index], 0) /
  predictions.length;
implementations.register_metric(metricReferenceJson, biasMetric);
const resolvedMetric = implementations.resolve_metric(metricReferenceJson);
if (resolvedMetric([1, 3], [2, 5]) !== 1.5) {
  throw new Error("JavaScript custom metric returned the wrong value");
}
if (implementations.size !== 2 || JSON.parse(implementations.descriptors_json()).length !== 2) {
  throw new Error("JavaScript local implementation registry has wrong descriptor coverage");
}
try {
  JSON.stringify(implementations);
  throw new Error("JavaScript local implementation registry was serialized");
} catch (error) {
  if (!String(error).includes("cannot be serialized")) {
    throw error;
  }
}
implementations.unregister_metric(metricReferenceJson);
implementations.unregister_loss(lossReferenceJson);
if (implementations.size !== 0) {
  throw new Error("JavaScript local implementation registry did not unregister callbacks");
}
implementations.free();

const artifact = JSON.parse(dagMl.compile_pipeline_dsl_artifact_json(dslJson));
if (!artifact.campaign_template) {
  throw new Error("compiled artifact is missing campaign_template");
}
const hostControllerSpecs = [
  {
    controller_id: "controller:wasm.smoke.transform",
    controller_version: "0.10.0",
    operator_kind: "transform",
  },
  {
    controller_id: "controller:wasm.smoke.model",
    controller_version: "0.10.0",
    operator_kind: "model",
    priority: 20,
  },
];
const derivedControllers = JSON.parse(
  dagMl.derive_controller_manifest_list_json(JSON.stringify(hostControllerSpecs)),
);
if (derivedControllers.length !== 2) {
  throw new Error("derived controller manifest list has wrong length");
}
dagMl.validate_controller_manifest_json(
  dagMl.derive_controller_manifest_json(JSON.stringify(hostControllerSpecs[0])),
);
dagMl.validate_controller_manifest_list_json(JSON.stringify(derivedControllers));

const runtimeGraph = {
  id: "javascript-local-loss-runtime",
  interface: { inputs: [], outputs: [] },
  nodes: [
    {
      id: "model:custom",
      kind: "model",
      operator: { type: "JavaScriptLocalLossProbe" },
      params: {},
      ports: { inputs: [], outputs: [] },
      metadata: {},
      seed_label: "javascript-local-loss",
    },
  ],
  edges: [],
  search_space_fingerprint: null,
  metadata: {},
};
const runtimeCampaign = {
  id: "campaign:javascript-local-loss-runtime",
  root_seed: 42,
};
const runtimeControllerManifest = JSON.parse(
  dagMl.derive_controller_manifest_json(
    JSON.stringify({
      controller_id: "controller:javascript-local",
      controller_version: "1.0.0",
      operator_kind: "model",
      added_capabilities: [
        "supports_configurable_loss",
        "supports_custom_loss",
        "supports_differentiable_loss",
      ],
      input_ports: [],
      output_ports: [],
    }),
  ),
);
const runtimePlanJson = dagMl.build_execution_plan_with_training_losses_json(
  "plan:javascript-local-loss-runtime",
  JSON.stringify(runtimeGraph),
  JSON.stringify(runtimeCampaign),
  JSON.stringify([runtimeControllerManifest]),
  JSON.stringify([localImplementations.training_loss_role]),
);
dagMl.validate_execution_plan_json(runtimePlanJson);
const runtimePlan = JSON.parse(runtimePlanJson);
if (runtimePlan.node_plans["model:custom"].training_losses.length !== 1) {
  throw new Error("native execution plan did not retain the JavaScript training loss");
}

const runtimeImplementations = new dagMl.LocalImplementationRegistry();
const runtimeLossCalls = [];
runtimeImplementations.register_loss(lossReferenceJson, (target, prediction) => {
  runtimeLossCalls.push([target, prediction]);
  return prediction >= target ? prediction - target : 2 * (target - prediction);
});
const runtimeResult = (controllerId, task, lossAttestations) => ({
  node_id: task.node_plan.node_id,
  lineage: {
    record_id: `lineage:javascript-local:${task.phase.toLowerCase()}`,
    run_id: task.run_id,
    node_id: task.node_plan.node_id,
    phase: task.phase,
    controller_id: controllerId,
    controller_version: task.node_plan.controller_version,
    variant_id: task.variant_id,
    fold_id: task.fold_id,
    branch_path: task.branch_path,
    input_lineage: [],
    artifact_refs: [],
    params_fingerprint: task.node_plan.params_fingerprint,
    data_model_shape_fingerprint: null,
    aggregation_policy_fingerprint: null,
    seed: null,
    unsafe_flags: [],
    metrics: {},
    loss_attestations: lossAttestations,
    early_stopping_records: [],
  },
});
const runtimeTaskSeeds = [];
const invokeRuntimeController = (controllerId, taskJson, exactSeed) => {
  const serializedSeedToken = /"seed":(\d+|null)\}$/u.exec(taskJson)?.[1];
  const serializedSeed = serializedSeedToken === "null" ? null : serializedSeedToken;
  if (exactSeed !== serializedSeed) {
    throw new Error("runtime bridge did not supply the exact native task seed");
  }
  runtimeTaskSeeds.push(exactSeed);
  const task = JSON.parse(taskJson);
  const binding = runtimeImplementations.bind_training_loss(taskJson, 0);
  let attestation;
  try {
    if (binding.invoke(4, 1.5) !== 5) {
      throw new Error(`runtime JavaScript custom loss returned the wrong value in ${task.phase}`);
    }
    attestation = JSON.parse(binding.required_attestation_json);
  } finally {
    binding.free();
  }
  return JSON.stringify(runtimeResult(controllerId, task, [attestation]));
};
for (const phase of ["FIT_CV", "REFIT"]) {
  const resultsJson = dagMl.execute_execution_plan_phase_json(
    runtimePlanJson,
    JSON.stringify([runtimeControllerManifest]),
    `run:javascript-local-runtime-${phase.toLowerCase()}`,
    42,
    phase,
    invokeRuntimeController,
  );
  const resultSeedToken = /"seed":(\d+|null),"unsafe_flags":/u.exec(resultsJson)?.[1];
  const resultSeed = resultSeedToken === "null" ? null : resultSeedToken;
  if (resultSeed !== runtimeTaskSeeds.at(-1)) {
    throw new Error(`runtime bridge did not preserve the exact task seed in ${phase}`);
  }
  const results = JSON.parse(resultsJson);
  if (
    results.length !== 1 ||
    results[0].lineage.loss_attestations.length !== 1 ||
    results[0].lineage.loss_attestations[0].phase !== phase
  ) {
    throw new Error(`scheduler did not validate the JavaScript loss lineage in ${phase}`);
  }
}
if (runtimeLossCalls.length !== 2) {
  throw new Error("runtime controller did not invoke the JavaScript custom loss in both phases");
}
if (runtimeTaskSeeds.length !== 2 || runtimeTaskSeeds.some((seed) => seed === null)) {
  throw new Error("runtime controller did not receive exact seeds in both phases");
}
const staleRuntimeControllerManifest = {
  ...runtimeControllerManifest,
  controller_version: "9.9.9",
};
try {
  dagMl.execute_execution_plan_phase_json(
    runtimePlanJson,
    JSON.stringify([staleRuntimeControllerManifest]),
    "run:javascript-local-runtime-stale-controller",
    42,
    "FIT_CV",
    invokeRuntimeController,
  );
  throw new Error("scheduler accepted a stale JavaScript controller manifest");
} catch (error) {
  if (!String(error).includes("does not match the trusted runtime manifest")) {
    throw error;
  }
}
try {
  dagMl.execute_execution_plan_phase_json(
    runtimePlanJson,
    JSON.stringify([runtimeControllerManifest]),
    "run:javascript-local-runtime-missing-attestation",
    42,
    "FIT_CV",
    (controllerId, taskJson) =>
      JSON.stringify(runtimeResult(controllerId, JSON.parse(taskJson), [])),
  );
  throw new Error("scheduler accepted a JavaScript controller without loss attestation");
} catch (error) {
  if (!String(error).includes("returned 0 loss attestations")) {
    throw error;
  }
}
runtimeImplementations.free();

const foldSet = {
  id: "cv.partition",
  sample_ids: ["s1", "s2", "s3"],
  folds: [
    {
      fold_id: "fold1",
      train_sample_ids: ["s1", "s2"],
      validation_sample_ids: ["s3"],
    },
    {
      fold_id: "fold0",
      train_sample_ids: ["s3"],
      validation_sample_ids: ["s2", "s1"],
    },
  ],
};
const foldSetJson = JSON.stringify(foldSet);
dagMl.validate_fold_set_json(foldSetJson);
dagMl.validate_fold_set_json(sharedFoldSetJson);
if (dagMl.fold_set_fingerprint_json(sharedFoldSetJson) !== SHARED_FOLD_SET_FINGERPRINT) {
  throw new Error("shared fold set fingerprint drifted");
}
const foldFingerprint = dagMl.fold_set_fingerprint_json(foldSetJson);
if (foldFingerprint.length !== 64) {
  throw new Error("fold set fingerprint is not a sha256 hex digest");
}
const reorderedFoldSet = {
  ...foldSet,
  sample_ids: [...foldSet.sample_ids].reverse(),
  folds: [...foldSet.folds].reverse(),
};
if (dagMl.fold_set_fingerprint_json(JSON.stringify(reorderedFoldSet)) !== foldFingerprint) {
  throw new Error("fold set fingerprint changed after irrelevant reordering");
}
if (!dagMl.dag_ml_version()) {
  throw new Error("dag_ml_version() returned an empty version");
}
try {
  dagMl.validate_graph_json('{"id":"","interface":{},"nodes":[],"edges":[]}');
  throw new Error("invalid graph JSON was accepted");
} catch (error) {
  const descriptor = parseErrorDescriptor(error);
  if (descriptor.category !== "validation" || descriptor.code !== "graph_validation") {
    throw new Error("WASM error descriptor taxonomy drifted");
  }
}
