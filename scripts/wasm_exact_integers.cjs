"use strict";

// These contracts carry u64 seeds. Node 20 has no JSON reviver context or
// JSON.rawJSON: retain only unsafe integer tokens as BigInt in this harness,
// then emit the original integer values when passing contracts back to WASM.
function integerMarker(value) {
  const strings = JSON.stringify(value, (_key, child) => typeof child === "bigint" ? null : child);
  let marker = "__dagml_exact_integer__";
  while (strings.includes(marker)) marker += "_";
  return marker;
}

function parseExactIntegers(text) {
  const marker = integerMarker(JSON.parse(text));
  const integers = [];
  // Match complete strings first, so digits in identifiers/escaped strings
  // cannot become numeric control values. JSON.parse validates the grammar.
  const protectedText = text.replace(/"(?:\\[\s\S]|[^"\\])*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/gu, token => {
    if (/^-?\d+$/u.test(token) && !Number.isSafeInteger(Number(token))) {
      const index = integers.push(BigInt(token)) - 1;
      return JSON.stringify(`${marker}${index}`);
    }
    return token;
  });
  return JSON.parse(protectedText, (_key, value) =>
    typeof value === "string" && value.startsWith(marker)
      ? integers[Number(value.slice(marker.length))] : value);
}

function stringifyExactIntegers(value) {
  const marker = integerMarker(value);
  const integers = [];
  const text = JSON.stringify(value, (_key, child) => {
    if (typeof child !== "bigint") return child;
    const index = integers.push(child.toString()) - 1;
    return `${marker}${index}`;
  });
  return text.replace(new RegExp(`"${marker}(\\d+)"`, "gu"), (_token, index) => integers[Number(index)]);
}

module.exports = {parseExactIntegers, stringifyExactIntegers};
