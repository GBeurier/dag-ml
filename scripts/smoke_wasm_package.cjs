#!/usr/bin/env node
"use strict";

// Import/version/export/package smoke. Full callback, CV/refit and HPO suites run locally.
const fs = require("fs");
const path = require("path");
if (!process.argv[2]) {
  throw new Error("Usage: node scripts/smoke_wasm_package.cjs <nodejs-package-directory>");
}
const pkgDir = path.resolve(process.argv[2]);
const dagMl = require(path.join(pkgDir, "dag_ml_wasm.js"));
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
  "execute_execution_plan_phase_u64_json",
  "execute_campaign_phase_u64_json",
  "execute_initial_full_refit_json",
  "replay_initial_full_refit_json",
  "validate_initial_full_refit_package_json",
  "initial_full_refit_predict_envelope_json",
  "select_stacking_producers_json",
  "fold_set_fingerprint_json",
  "host_hpo_search_json",
  "host_hpo_search_parallel_json",
  "host_hpo_evaluate_worker_task_json",
  "host_hpo_evaluate_worker_fold_json",
  "recover_host_hpo_checkpoint_json",
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

const manifest = JSON.parse(dagMl.contract_manifest_json());
if (manifest.crate !== "dag-ml" || manifest.version !== dagMl.dag_ml_version()) {
  throw new Error("WASM contract manifest crate/version mismatch");
}
assertPackageMetadata(manifest.version);
for (const name of REQUIRED_DTS_EXPORTS) {
  if (typeof dagMl[name] !== "function") {
    throw new Error(`WASM JavaScript export is missing ${name}()`);
  }
}
console.log(`dag-ml-wasm ${manifest.version}: package/import/types smoke passed`);
