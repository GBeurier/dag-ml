"use strict";

const path = require("path");
const { parentPort, workerData } = require("worker_threads");
const dagMl = require(path.join(workerData.pkgDir, "dag_ml_wasm.js"));
const { ridgeNodeResult } = require("./hpo_ridge_operator.cjs");
const strictJson = import("../bindings/js/n4m_multimodal_controller.mjs").then(module => module.strictJson);

const controller = (controllerId, taskJson) => ridgeNodeResult(controllerId, taskJson, workerData.folds, "js-hpo-worker");

parentPort.on("message", async (taskJson) => {
  try {
    const packet = JSON.parse(taskJson);
    const task = packet.kind === "fold" || packet.kind === "complete"
      ? (await strictJson)(taskJson, "task") : taskJson;
    const result = packet.kind === "fold"
      ? dagMl.host_hpo_evaluate_worker_fold_json(
        task, packet.fold_index, workerData.manifests,
        workerData.envelope, workerData.request, controller,
      )
      : dagMl.host_hpo_evaluate_worker_task_json(
        task,
        workerData.manifests, workerData.envelope, workerData.request, controller,
      );
    parentPort.postMessage({ result });
  } catch (error) {
    parentPort.postMessage({ error: String(error) });
  }
});
