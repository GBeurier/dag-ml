#!/usr/bin/env node
import fs from "fs";
import path from "path";
import { fileURLToPath, pathToFileURL } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(__dirname, "..");
const dagMlPkgDir = path.resolve(
  process.argv[2] || path.join(repo, "target", "wasm-web", "dag-ml-wasm"),
);
const dagMlDataPkgDir = path.resolve(
  process.argv[3] ||
    path.join(repo, "..", "dag-ml-data", "target", "wasm-web", "dag-ml-data-wasm"),
);
const dagMlDataRepo = path.resolve(process.argv[4] || path.join(repo, "..", "dag-ml-data"));

const SHARED_FOLD_SET_FINGERPRINT =
  "54d3185d6c628ef0df848828a8d8ae650222a283a78bbd3ab3bc2256f222c05c";

async function loadWasmPackage(pkgDir, jsName, wasmName) {
  const mod = await import(pathToFileURL(path.join(pkgDir, jsName)).href);
  const wasmBytes = fs.readFileSync(path.join(pkgDir, wasmName));
  mod.initSync({ module: wasmBytes });
  return mod;
}

function readText(root, relative) {
  return fs.readFileSync(path.join(root, relative), "utf8");
}

function requireCondition(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function assertPackageMetadata(pkgDir, expected) {
  const packageJson = JSON.parse(fs.readFileSync(path.join(pkgDir, "package.json"), "utf8"));
  requireCondition(
    packageJson.name === expected.packageName,
    `${expected.packageName} package.json has wrong package name`,
  );
  requireCondition(
    packageJson.version === expected.version,
    `${expected.packageName} package.json version does not match manifest`,
  );
  requireCondition(
    packageJson.main === expected.jsName || packageJson.module === expected.jsName,
    `${expected.packageName} package.json does not point to ${expected.jsName}`,
  );
  requireCondition(
    packageJson.types === expected.dtsName,
    `${expected.packageName} package.json does not point to ${expected.dtsName}`,
  );
  for (const filename of [expected.jsName, expected.wasmName, expected.dtsName]) {
    requireCondition(
      fs.existsSync(path.join(pkgDir, filename)),
      `${expected.packageName} package is missing ${filename}`,
    );
  }
}

const dagMl = await loadWasmPackage(dagMlPkgDir, "dag_ml_wasm.js", "dag_ml_wasm_bg.wasm");
const dagMlData = await loadWasmPackage(
  dagMlDataPkgDir,
  "dag_ml_data_wasm.js",
  "dag_ml_data_wasm_bg.wasm",
);

const dagManifest = JSON.parse(dagMl.contract_manifest_json());
const dataManifest = JSON.parse(dagMlData.contract_manifest_json());
requireCondition(dagManifest.crate === "dag-ml", "dag-ml manifest has wrong crate name");
requireCondition(
  dataManifest.crate === "dag-ml-data",
  "dag-ml-data manifest has wrong crate name",
);
requireCondition(
  dagManifest.version === dagMl.dag_ml_version(),
  "dag-ml manifest version does not match WASM export",
);
requireCondition(
  dataManifest.version === dagMlData.dag_ml_data_version(),
  "dag-ml-data manifest version does not match WASM export",
);
assertPackageMetadata(dagMlPkgDir, {
  packageName: "dag-ml-wasm",
  version: dagManifest.version,
  jsName: "dag_ml_wasm.js",
  wasmName: "dag_ml_wasm_bg.wasm",
  dtsName: "dag_ml_wasm.d.ts",
});
assertPackageMetadata(dagMlDataPkgDir, {
  packageName: "dag-ml-data-wasm",
  version: dataManifest.version,
  jsName: "dag_ml_data_wasm.js",
  wasmName: "dag_ml_data_wasm_bg.wasm",
  dtsName: "dag_ml_data_wasm.d.ts",
});
requireCondition(
  dagManifest.shared.fold_set_fixture_fingerprint ===
    dataManifest.shared.fold_set_fixture_fingerprint,
  "shared fold set fingerprint differs between WASM packages",
);
requireCondition(
  dagManifest.shared.fold_set_fixture_fingerprint === SHARED_FOLD_SET_FINGERPRINT,
  "shared fold set fingerprint drifted",
);
for (const name of ["compile_pipeline_dsl_artifact_json", "build_execution_plan_json"]) {
  requireCondition(dagManifest.wasm_exports.includes(name), `dag-ml manifest misses ${name}`);
}
for (const name of ["plan_model_input_json", "build_coordinator_data_plan_envelope_json"]) {
  requireCondition(
    dataManifest.wasm_exports.includes(name),
    `dag-ml-data manifest misses ${name}`,
  );
}

const dataFixtureRoot = path.join(dagMlDataRepo, "examples", "fixtures", "oof_campaign");
const schemaJson = readText(dataFixtureRoot, "schema_nirs4all_lite_contract.json");
const modelInputJson = readText(dataFixtureRoot, "model_input_tabular_numeric.json");
const adapterRegistryJson = readText(dataFixtureRoot, "adapter_registry_signal_to_tabular.json");
const relationsJson = readText(dataFixtureRoot, "sample_relations_grouped_augmented.json");
const dataPlanRequestJson = JSON.stringify({ id: "nir-to-tabular", source_ids: ["nir"] });

dagMlData.validate_dataset_schema_json(schemaJson);
dagMlData.validate_model_input_spec_json(modelInputJson);
dagMlData.validate_adapter_registry_json(adapterRegistryJson);
dagMlData.validate_sample_relation_table_json(relationsJson);
const dataPlanJson = dagMlData.plan_model_input_json(
  schemaJson,
  modelInputJson,
  adapterRegistryJson,
  dataPlanRequestJson,
);
dagMlData.validate_data_plan_json(dataPlanJson);
const envelopeJson = dagMlData.build_coordinator_data_plan_envelope_json(
  schemaJson,
  dataPlanJson,
  relationsJson,
);
dagMlData.validate_coordinator_data_plan_envelope_json(envelopeJson);
const envelope = JSON.parse(envelopeJson);
requireCondition(envelope.plan.id === "nir-to-tabular", "data envelope has wrong plan id");
requireCondition(
  typeof envelope.relation_fingerprint === "string" && envelope.relation_fingerprint.length === 64,
  "data envelope is missing relation fingerprint",
);

const dslJson = readText(repo, "examples/pipeline_dsl_nirs4all_compat.json");
const controllerManifestsJson = readText(repo, "examples/controller_manifests.json");
dagMl.validate_pipeline_dsl_json(dslJson);
dagMl.validate_controller_manifest_list_json(controllerManifestsJson);
const artifact = JSON.parse(dagMl.compile_pipeline_dsl_artifact_json(dslJson));
dagMl.validate_graph_json(JSON.stringify(artifact.graph));
dagMl.validate_campaign_json(JSON.stringify(artifact.campaign_template));
requireCondition(artifact.graph.nodes.length > 0, "compiled graph contains no nodes");
requireCondition(
  artifact.campaign_template.split_invocation,
  "compiled campaign template is missing split invocation",
);
const executionPlanJson = dagMl.build_execution_plan_json(
  "plan:wasm.integration",
  JSON.stringify(artifact.graph),
  JSON.stringify(artifact.campaign_template),
  controllerManifestsJson,
);
dagMl.validate_execution_plan_json(executionPlanJson);
const executionPlan = JSON.parse(executionPlanJson);
requireCondition(Object.keys(executionPlan.node_plans).length > 0, "execution plan has no nodes");
requireCondition(executionPlan.variants.length > 0, "execution plan has no variants");
