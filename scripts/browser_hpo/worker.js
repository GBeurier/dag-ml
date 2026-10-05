import init, * as dagMl from "/pkg/dag_ml_wasm.js";
import {ridgeNodeResult} from "/ridge_operator.js";

const ready = init();

// The embedded native plan carries u64 seeds. Keep their original JSON
// tokens when extracting the task instead of rounding through Number.
function parseExactIntegers(text) {
  return JSON.parse(text, (_key, value, context) => {
    if (typeof value !== "number" || Number.isSafeInteger(value)) return value;
    if (!context || typeof JSON.rawJSON !== "function") {
      throw new Error("Browser HPO worker requires lossless native numeric JSON transport");
    }
    return /^-?\d+$/u.test(context.source) ? JSON.rawJSON(context.source) : value;
  });
}

self.onmessage = async event => {
  try {
    await ready;
    const {taskJson, manifests, envelope, request, folds} = event.data;
    const packet = parseExactIntegers(taskJson);
    const controller = (controllerId, nodeTaskJson) => ridgeNodeResult(controllerId, nodeTaskJson, folds);
    const result = packet.kind === "fold"
      ? dagMl.host_hpo_evaluate_worker_fold_json(
        JSON.stringify(packet.task), packet.fold_index, manifests, envelope, request, controller)
      : dagMl.host_hpo_evaluate_worker_task_json(
        packet.kind === "complete" ? JSON.stringify(packet.task) : taskJson,
        manifests, envelope, request, controller);
    self.postMessage({result});
  } catch (error) {
    self.postMessage({error: String(error.stack || error)});
  }
};
