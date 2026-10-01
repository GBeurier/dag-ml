#!/usr/bin/env node
/** Fresh-process consumer for a genuine five-model Core Archive V2. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const [packagePath, methodsPath, coreModulePath, archivePath, capturePath] = process.argv.slice(2);
const packageDir = path.resolve(packagePath);
const metadata = JSON.parse(fs.readFileSync(path.join(packageDir, "package.json"), "utf8"));
const dagMl = metadata.type === "module" || metadata.module ?
  await import(pathToFileURL(path.join(packageDir, "dag_ml_wasm.js")).href) :
  createRequire(import.meta.url)(path.join(packageDir, "dag_ml_wasm.js"));
if (metadata.type === "module" || metadata.module)
  dagMl.initSync({ module: fs.readFileSync(path.join(packageDir, "dag_ml_wasm_bg.wasm")) });
const methods = await import(pathToFileURL(path.join(path.resolve(methodsPath), "index.js")).href);
await methods.loadModule();
const core = await import(pathToFileURL(path.resolve(coreModulePath)).href);
const { N4mWasmRegressionController } = await import(
  pathToFileURL(path.join(packageDir, "n4m_controller.mjs")).href);
const capture = JSON.parse(fs.readFileSync(capturePath, "utf8"));
const archiveBytes = new Uint8Array(fs.readFileSync(archivePath));
const loaded = await core.readPortableArchiveV2(archiveBytes);
const packageJson = new TextDecoder("utf-8", { fatal: true }).decode(
  loaded.members["dagml/portable_predictor_package.json"]);
const packageValue = JSON.parse(packageJson);
const wireMembers = Object.fromEntries(Object.entries(loaded.members).map(
  ([member, bytes]) => [member, Array.from(bytes)]));
const validate = (manifest, members) => JSON.parse(dagMl.validate_archive_v2_portable_payloads_json(
  JSON.stringify(manifest), packageJson, JSON.stringify(members)));
assert.deepEqual(validate(loaded.manifest, wireMembers), { valid: true });
assert.equal(packageValue.package_fingerprint, JSON.parse(capture.packageJson).package_fingerprint);
assert.equal(loaded.manifest.payloads.methods.n4mm.length, 0);
assert.equal(loaded.manifest.payloads.methods.role_pipelines.length, 5);
assert.equal(Object.keys(loaded.members).length, 11);
assert.equal(packageValue.artifact_bindings.length, 5);

const digest = bytes => createHash("sha256").update(bytes).digest("hex");
const nodeIds = packageValue.predictor_node_ids;
const operators = Object.fromEntries(nodeIds.map(nodeId =>
  [nodeId, { type: "n4m:models.regularized.ridge" }]));
const sampleId = capture.replay.outputs[0].predictions[0].sample_ids[0];
const makeController = () => new N4mWasmRegressionController({ methods, operators, digest,
  resolveFeatures: ({ view }) => {
    assert.deepEqual(view.sample_ids, [sampleId]);
    assert.equal(view.source_ids.length, 1);
    const row = capture.heldoutRows[view.source_ids[0]];
    assert.ok(row, "Only the explicitly supplied current source may be read");
    return { sampleIds: [sampleId], matrix: { data: Float64Array.from(row), rows: 1, cols: row.length } };
  },
  resolveTargets: () => { throw new Error("Archive prediction cannot read targets"); },
});
const controller = makeController();
// Trust comes from this installed controller, independently of archived bytes.
const trustedManifest = controller.manifest(dagMl);
assert.deepEqual(trustedManifest, capture.manifest);
const operations = [];
const invoke = (manifests, callback) => dagMl.replay_training_package_json(packageJson,
  capture.replayRequestJson, JSON.stringify(capture.predictEnvelopes), JSON.stringify(manifests),
  "outcome:archive.fresh-consumer", "run:archive.fresh-consumer", callback);
let replay;
const originalFit = methods.RolePipeline.prototype.fit;
methods.RolePipeline.prototype.fit = () => { throw new Error("Archive consumer must never fit"); };
try {
  replay = JSON.parse(invoke([trustedManifest], (id, json, seed) => {
    const task = JSON.parse(json);
    const operation = task.operation ?? task.phase;
    assert.ok(!["FIT", "CV", "SELECT", "REFIT"].includes(operation));
    operations.push({ operation, node: task.node_plan?.node_id ?? task.request?.node_id });
    return controller.callback(id, json, seed);
  }));
  assert.equal(controller.models.size, 0);
  let callbacks = 0;
  const badTrust = { ...trustedManifest, controller_version: "999.0.0" };
  assert.throws(() => invoke([badTrust], () => { callbacks++; }));
  assert.equal(callbacks, 0);
  const modified = structuredClone(wireMembers);
  modified[loaded.manifest.payloads.methods.role_pipelines[0].member_path][0] ^= 1;
  assert.throws(() => validate(loaded.manifest, modified));
  const missing = structuredClone(wireMembers);
  delete missing[loaded.manifest.payloads.methods.role_pipelines[0].member_path];
  assert.throws(() => validate(loaded.manifest, missing));
  const aliased = structuredClone(loaded.manifest);
  aliased.payloads.methods.role_pipelines[1].member_path = aliased.payloads.methods.role_pipelines[0].member_path;
  assert.throws(() => validate(aliased, wireMembers));
  const failed = makeController();
  try {
    assert.throws(() => invoke([trustedManifest], (id, json, seed) => {
      if (JSON.parse(json).phase === "PREDICT") throw new Error("Injected archive prediction failure");
      return failed.callback(id, json, seed);
    }));
    assert.equal(failed.models.size, 0, "Failure must release every hydrated model");
  } finally { failed.close(); }
} finally {
  methods.RolePipeline.prototype.fit = originalFit;
  controller.close();
}
assert.equal(operations.filter(task => task.operation === "hydrate_artifact_payload").length, 5);
assert.equal(operations.filter(task => task.operation === "release_hydrated_artifact_payload").length, 5);
assert.deepEqual(operations.filter(task => task.operation === "PREDICT").map(task => task.node).sort(),
  [...nodeIds].sort());
const prediction = replay.outputs[0].predictions[0];
assert.deepEqual(prediction.sample_ids, [sampleId]);
const expected = capture.replay.outputs[0].predictions[0].values.flat();
const actual = prediction.values.flat();
assert.equal(actual.length, expected.length);
actual.forEach((value, index) => assert.ok(Number.isFinite(value) && Math.abs(value - expected[index]) < 1e-9));
console.log(JSON.stringify({ archiveId: loaded.archiveId, archiveSha256: digest(archiveBytes),
  coreModule: path.resolve(coreModulePath), dagmlPackage: packageDir,
  packageFingerprint: packageValue.package_fingerprint, modelCount: 5, memberCount: 11,
  fits: 0, hydrations: 5, predictions: 5, releases: 5, currentSourcesOnly: true,
  trustedManifestsChecked: true, tamperAndMissingRefused: true, failedReplayReleasedStates: true }));
