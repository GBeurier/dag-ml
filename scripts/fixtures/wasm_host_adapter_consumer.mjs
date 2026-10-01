// Run from the extracted package's consumer, without installing Methods.
import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import { N4mWasmRegressionController } from "dag-ml-wasm/n4m-controller";
import { N4mWasmHostOptimizer } from "dag-ml-wasm/n4m-optimizer";
import * as dagMl from "dag-ml-wasm";

if (typeof dagMl.initSync === "function") {
  const require = createRequire(import.meta.url);
  dagMl.initSync({ module: fs.readFileSync(require.resolve("dag-ml-wasm/dag_ml_wasm_bg.wasm")) });
}
assert.equal(typeof N4mWasmRegressionController, "function");
assert.equal(typeof N4mWasmHostOptimizer, "function");
for (const name of ["training_data_identity_json", "sample_relation_set_fingerprint_json",
  "sign_training_request_json", "execute_training_json", "attach_predict_cohort_to_envelope_json",
  "sign_training_replay_request_json", "replay_training_package_json"])
  assert.equal(typeof dagMl[name], "function", "Missing packaged native training export: " + name);
assert.equal(dagMl.dag_ml_version(), JSON.parse(dagMl.contract_manifest_json()).version);
console.log("PACKAGED_HOST_ADAPTER_IMPORTS_OK", dagMl.dag_ml_version());
