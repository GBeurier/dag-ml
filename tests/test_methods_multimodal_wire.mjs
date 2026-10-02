import assert from "node:assert/strict";
import test from "node:test";
import {multimodalManifest, strictJson, utf8Cell} from "../bindings/js/n4m_multimodal_controller.mjs";

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
