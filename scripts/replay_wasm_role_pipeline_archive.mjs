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
// Core's generic native reader admitted the container and its selected scientific closure. `package`
// and `trainingOutcome` are the exact serialized root bytes; there is no physical manifest replay map.
const utf8 = new TextEncoder();
assert.equal(loaded.view.profile, "native-methods-package");
assert.equal(loaded.view.package_schema_version, 2);
assert.equal(loaded.archiveSha256, createHash("sha256").update(archiveBytes).digest("hex"));
assert.ok(Buffer.from(loaded.package).equals(Buffer.from(utf8.encode(capture.packageJson))),
  "The native reader must return the exact original package bytes");
assert.ok(Buffer.from(loaded.trainingOutcome).equals(Buffer.from(utf8.encode(capture.outcomeJson))),
  "The native reader must return the exact original TrainingOutcome bytes");
assert.ok(Buffer.from(loaded.members[loaded.view.package_member]).equals(Buffer.from(loaded.package)));
const packageJson = new TextDecoder("utf-8", { fatal: true }).decode(loaded.package);
const packageValue = JSON.parse(packageJson);
assert.equal(packageValue.package_fingerprint, JSON.parse(capture.packageJson).package_fingerprint);
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
  // Core refuses a corrupted selected member or a truncated container before any scientific use.
  const stateMember = Object.entries(loaded.members)
    .filter(([member]) => member !== loaded.view.package_member)
    .sort(([, left], [, right]) => right.length - left.length)[0][1];
  const stateOffset = Buffer.from(archiveBytes).indexOf(Buffer.from(stateMember));
  assert.ok(stateOffset >= 0, "Stored physical members must appear verbatim in the container");
  const corrupted = Uint8Array.from(archiveBytes);
  corrupted[stateOffset] ^= 1;
  await assert.rejects(() => core.readPortableArchiveV2(corrupted));
  await assert.rejects(() => core.readPortableArchiveV2(archiveBytes.slice(0, archiveBytes.length - 1)));
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
  packageFingerprint: packageValue.package_fingerprint, modelCount: 5, memberCount: Object.keys(loaded.members).length,
  profile: loaded.view.profile, packageSchemaVersion: loaded.view.package_schema_version,
  fits: 0, hydrations: 5, predictions: 5, releases: 5, currentSourcesOnly: true,
  trustedManifestsChecked: true, tamperAndTruncationRefused: true, failedReplayReleasedStates: true }));
