import init, * as dagMl from "/pkg/dag_ml_wasm.js";
import {ridgeNodeResult} from "/ridge_operator.js";
import {strictJson} from "/n4m_multimodal_controller.mjs";

const ready = init();

self.onmessage = async event => {
  try {
    await ready;
    const {taskJson, manifests, envelope, request, folds} = event.data;
    const packet = JSON.parse(taskJson);
    const task = packet.kind === "fold" || packet.kind === "complete" ? strictJson(taskJson, "task") : taskJson;
    const controller = (controllerId, nodeTaskJson) => ridgeNodeResult(controllerId, nodeTaskJson, folds);
    const result = packet.kind === "fold"
      ? dagMl.host_hpo_evaluate_worker_fold_json(
        task, packet.fold_index, manifests, envelope, request, controller)
      : dagMl.host_hpo_evaluate_worker_task_json(
        task,
        manifests, envelope, request, controller);
    self.postMessage({result});
  } catch (error) {
    self.postMessage({error: String(error.stack || error)});
  }
};
