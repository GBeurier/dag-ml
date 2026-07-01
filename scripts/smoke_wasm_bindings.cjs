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
  "compile_pipeline_dsl_artifact_json",
  "compile_pipeline_dsl_graph_json",
  "contract_manifest_json",
  "dag_ml_version",
  "derive_controller_manifest_json",
  "derive_controller_manifest_list_json",
  "fold_set_fingerprint_json",
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
if (manifest.shared.fold_set_fixture_fingerprint !== SHARED_FOLD_SET_FINGERPRINT) {
  throw new Error("contract manifest shared fold fingerprint drifted");
}
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
