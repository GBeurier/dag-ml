import assert from "node:assert/strict";
import test from "node:test";
import {createHash} from "node:crypto";
import {boundedIdentifier, multimodalManifest, strictJson, utf8Cell} from "../bindings/js/n4m_multimodal_controller.mjs";

const digest = bytes => createHash("sha256").update(bytes).digest("hex");

test("native identifier boundary preserves short lineage and artifact identities", () => {
  for (const prefix of ["lineage:methods-multimodal", "artifact:methods.multimodal"]) {
    const coordinate = "r".repeat(127 - prefix.length), original = `${prefix}:${coordinate}`;
    assert.equal(new TextEncoder().encode(original).length, 128);
    assert.equal(boundedIdentifier([prefix, coordinate], digest), original);
    assert.ok(new TextEncoder().encode(boundedIdentifier([prefix, coordinate + "r"], digest)).length <= 128);
  }
});

test("long native identities match Python vectors and preserve complete coordinate scope", () => {
  const prefix = "lineage:methods-multimodal", run = "run:" + "r".repeat(100);
  const first = boundedIdentifier([prefix, run, "model:a", "PREDICT", "base", "full"], digest);
  const second = boundedIdentifier([prefix, run + ":model", "a", "PREDICT", "base", "full"], digest);
  assert.equal(first, `${prefix}:22407a3baa33ed60aad13b2a8bcde292d69e0e1cfd2f17c0c5e145bed261a681`);
  assert.equal(second, `${prefix}:566707755525c2a470aa50d6b58d7c3137a885f884d1e6b740240cbc0bb473bd`);
  assert.notEqual(first, second);
  assert.equal(boundedIdentifier(["artifact:methods.multimodal", run, "model:a", "base", "refit"], digest),
    "artifact:methods.multimodal:4a2b4526eefcf65d51ca84386b191702af8245079ff0f47ee9b19957f1a9d96b");
});

test("raw category validation preserves empty strings and enforces UTF-8 byte bounds", () => {
  assert.equal(utf8Cell("", 0), true);
  assert.equal(utf8Cell("µ", 2), true);
  assert.equal(utf8Cell("µ", 1), false);
  assert.equal(utf8Cell("category", 8), true);
  assert.equal(utf8Cell("category", 7), false);
  for(const value of [null, undefined, 0, false, [], {}]) assert.equal(utf8Cell(value, 1048576), false);
});

test("all trusted producer manifests retain raw named sources with sample identity alignment", () => {
  for (const host of ["python", "wasm", "r", "octave"]) {
    const requirements = multimodalManifest(host).data_requirements;
    assert.deepEqual(requirements.default_fusion, {mode:"dict_by_source",alignment:"sample_id",adapter_id:null,params:{}});
    assert.equal(requirements.ports[0].multi_source, true);
    assert.equal(requirements.ports[0].rank, null);
    assert.deepEqual(requirements.ports[0].accepted_representations, ["feature_block_set"]);
  }
});

test("raw task member extraction preserves null and u64 without confusing the frame type", () => {
  for(const seed of ["null","18446744073709551615"]) {
    const task=`{"seed":${seed},"data_views":{"input:/x.v1":{"sample_ids":["sample:a"]}}}`;
    const frame=`{"type":"task","schema_version":1,"task":${task},"tail":"task"}`;
    assert.equal(strictJson(frame,"task"),task);
    assert.equal(strictJson(strictJson(frame,"task"),"seed"),seed);
  }
  assert.throws(()=>strictJson('{"type":"task","task":{},"task":{}}',"task"),/Duplicate/);
  assert.throws(()=>strictJson('{"type":"task"}',"task"),/Missing raw JSON member/);
});

test("native numeric contract transport preserves integer and binary64 token identity", () => {
  const raw='{"generation":{"choices":[1,1.0,0.5]},"seed":18446744073709551615}';
  const transported=JSON.parse(raw,(key,value,context)=>typeof value==="number"?JSON.rawJSON(context.source):value);
  assert.equal(JSON.stringify(transported),raw);
  assert.equal(strictJson(JSON.stringify(transported),"generation"),'{"choices":[1,1.0,0.5]}');
});
