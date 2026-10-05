#!/usr/bin/env node
/** Include optional host adapters in both wasm-pack package targets. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const packageDir = path.resolve(process.argv[2] ?? "");
if (!process.argv[2]) throw new Error("Usage: stage_wasm_host_adapters.mjs <wasm-pack-dir>");
const metadataPath = path.join(packageDir, "package.json");
const metadata = JSON.parse(fs.readFileSync(metadataPath, "utf8"));
if (metadata.name !== "dag-ml-wasm" || metadata.types !== "dag_ml_wasm.d.ts" ||
    (metadata.main !== "dag_ml_wasm.js" && metadata.module !== "dag_ml_wasm.js")) {
  throw new Error("Host adapters require a generated dag-ml-wasm package");
}
for (const name of ["dag_ml_wasm.js", "dag_ml_wasm_bg.wasm", "dag_ml_wasm.d.ts"]) {
  if (!fs.statSync(path.join(packageDir, name)).isFile()) throw new Error("Missing generated WASM file");
}
const extras = ["n4m_controller.mjs", "n4m_controller.d.ts", "n4m_hpo_optimizer.mjs", "n4m_hpo_optimizer.d.ts", "n4m_multimodal_controller.mjs", "multimodal_archive.mjs", "multimodal_dataset_replay.mjs"];
for (const name of extras) fs.copyFileSync(path.join(repo, "bindings/js", name), path.join(packageDir, name));
metadata.files = [...new Set([...metadata.files, ...extras])];
metadata.exports = {
  ...metadata.exports,
  ".": { types: "./dag_ml_wasm.d.ts", default: "./dag_ml_wasm.js" },
  "./n4m-controller": { types: "./n4m_controller.d.ts", import: "./n4m_controller.mjs" },
  "./n4m-optimizer": { types: "./n4m_hpo_optimizer.d.ts", import: "./n4m_hpo_optimizer.mjs" },
  "./multimodal_dataset_replay": { import: "./multimodal_dataset_replay.mjs" },
  // Preserve existing direct access to generated JS/WASM/package files.
  "./*": "./*",
};
metadata.peerDependencies = { ...metadata.peerDependencies, "@nirs4all/methods": "^1.2.1" };
metadata.peerDependenciesMeta = { ...metadata.peerDependenciesMeta, "@nirs4all/methods": { optional: true } };
fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2) + "\n");
console.log("Staged optional Methods controllers and HPO optimizer: " + metadata.version);
