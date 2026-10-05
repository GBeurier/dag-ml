"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {parseExactIntegers, stringifyExactIntegers} = require("./wasm_exact_integers.cjs");

test("u64 and adjacent unsafe integers survive native JSON and structuredClone", () => {
  const text = '{"safe":9007199254740991,"a":9007199254740992,"b":9007199254740993,"max":18446744073709551615,"signed":-9223372036854775808}';
  const parsed = parseExactIntegers(text);
  assert.equal(parsed.safe, Number.MAX_SAFE_INTEGER);
  assert.equal(parsed.a, 9007199254740992n);
  assert.equal(parsed.b, 9007199254740993n);
  assert.notEqual(parsed.a, parsed.b);
  assert.equal(parsed.max, 18446744073709551615n);
  assert.equal(parsed.signed, -9223372036854775808n);
  assert.equal(stringifyExactIntegers(structuredClone(parsed)), text);
});

test("numeric strings, escaped strings and marker-like values keep their types", () => {
  const text = String.raw`{"seed":18446744073709551615,"id":"18446744073709551615","escaped":"quote\" and slash\\ 9007199254740993","marker":"\u005f_dagml_exact_integer__0","__dagml_exact_integer___1":"value","items":["__dagml_exact_integer__1",9007199254740993]}`;
  const parsed = parseExactIntegers(text);
  assert.equal(parsed.id, "18446744073709551615");
  assert.equal(parsed.escaped, 'quote" and slash\\ 9007199254740993');
  assert.equal(parsed.marker, "__dagml_exact_integer__0");
  assert.equal(parsed.items[0], "__dagml_exact_integer__1");
  assert.equal(parsed.items[1], 9007199254740993n);
  assert.deepEqual(parseExactIntegers(stringifyExactIntegers(parsed)), parsed);
});

test("ordinary floats, arrays and malformed JSON retain JSON semantics", () => {
  const parsed = parseExactIntegers('{"fraction":1.25,"largeFloat":1e30,"explicitFloat":9007199254740993.0,"items":[true,null,"9",-7]}');
  assert.equal(typeof parsed.largeFloat, "number");
  assert.equal(typeof parsed.explicitFloat, "number");
  assert.equal(parsed.fraction, 1.25);
  assert.deepEqual(JSON.parse(stringifyExactIntegers(parsed)), parsed);
  assert.throws(() => parseExactIntegers('{"seed":01}'), SyntaxError);
  assert.throws(() => parseExactIntegers('{"seed":18446744073709551615,}'), SyntaxError);
});
