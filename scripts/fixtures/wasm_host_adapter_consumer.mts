// Compile against the extracted npm tarball, including its optional peer types.
import {
  N4mWasmRegressionController,
  type MethodsRegressionControllerOptions,
} from "dag-ml-wasm/n4m-controller";
import {
  N4mWasmHostOptimizer,
  type N4mWasmHpoOptions,
} from "dag-ml-wasm/n4m-optimizer";
import * as methods from "@nirs4all/methods";
import * as dagMl from "dag-ml-wasm";

// Generated declarations must expose the native training and replay entry points.
const nativeTraining: typeof dagMl.execute_training_json = dagMl.execute_training_json;
const nativeReplay: typeof dagMl.replay_training_package_json = dagMl.replay_training_package_json;
void nativeTraining;
void nativeReplay;

const controllerOptions: MethodsRegressionControllerOptions = {
  methods,
  operators: { "model:ridge": { type: "n4m:models.regularized.ridge" } },
  resolveFeatures: ({ view }) => ({
    sampleIds: view.sample_ids,
    matrix: { data: new Float64Array(view.sample_ids.length), rows: view.sample_ids.length, cols: 1 },
  }),
  resolveTargets: ({ sampleIds }) => ({
    sampleIds,
    matrix: { data: new Float64Array(sampleIds.length), rows: sampleIds.length, cols: 1 },
  }),
};
const optimizerOptions: N4mWasmHpoOptions = {
  Optimizer: methods.Optimizer,
  dagMl,
  space: { alpha: { kind: "float", low: 0.1, high: 1 } },
  options: { sampler: "sobol", metric: "rmse", direction: "minimize", pruner: "none", seed: 7 },
  objective: {},
  persist: snapshot => { console.log(snapshot.schema); },
};
new N4mWasmRegressionController(controllerOptions).close();
new N4mWasmHostOptimizer(optimizerOptions).close();
