"use strict";

const path = require("path");
const { parentPort, workerData } = require("worker_threads");
const dagMl = require(path.join(workerData.pkgDir, "dag_ml_wasm.js"));
const { ridgeNodeResult } = require("./hpo_ridge_operator.cjs");

const controller = (controllerId, taskJson) => ridgeNodeResult(controllerId, taskJson, workerData.folds, "js-hpo-worker");

parentPort.on("message", (taskJson) => {
  try {
    const packet = JSON.parse(taskJson);
    const result = packet.kind === "fold"
      ? dagMl.host_hpo_evaluate_worker_fold_json(
        JSON.stringify(packet.task), packet.fold_index, workerData.manifests,
        workerData.envelope, workerData.request, controller,
      )
      : dagMl.host_hpo_evaluate_worker_task_json(
        packet.kind === "complete" ? JSON.stringify(packet.task) : taskJson,
        workerData.manifests, workerData.envelope, workerData.request, controller,
      );
    parentPort.postMessage({ result });
  } catch (error) {
    parentPort.postMessage({ error: String(error) });
  }
});
