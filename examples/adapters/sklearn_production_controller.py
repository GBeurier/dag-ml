#!/usr/bin/env python3
"""Production sklearn process adapter for dag-ml.

Promotes the smoke adapter (`sklearn_process_controller.py`) to a
controller-ready surface:

* `operator_selectors` dispatch over `sklearn.preprocessing`,
  `sklearn.linear_model`, `sklearn.ensemble` and `sklearn.decomposition`,
  resolved through a single registry rather than a hard-coded
  StandardScaler+Ridge pipeline.
* Disk-backed artifact persistence through `joblib.dump`/`joblib.load`,
  so PREDICT can replay a model fitted in REFIT regardless of which
  persistent worker handled REFIT.
* Side-by-side with the smoke: the smoke remains the synthetic-data
  fixture used by `cli_contracts.rs`; this adapter advertises a
  distinct `adapter_id` and an extra `sklearn_production` capability.

Structured error frames, timeout/resource limits and a controller
manifest are still pending (Slices F.2 and F.3).
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import sys
from typing import Any

PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION = 1
PROCESS_ADAPTER_PROTOCOL = "dag-ml-process-adapter"
PROCESS_ADAPTER_MODES = ["one_shot", "jsonl"]
PROCESS_ADAPTER_CAPABILITIES = [
    "control_frames_v1",
    "node_task_json_v1",
    "node_result_json_v1",
    "parallel_invocation_v1",
    "persistent_workers",
    "worker_env",
    "stateful_refit_artifacts",
    "sklearn_production",
]
PROCESS_ADAPTER_FRAME_SCHEMA_VERSION = 1
ADAPTER_ID = "dag-ml-sklearn-production-controller"
ADAPTER_PLUGIN = "dagml.sklearn_production"
ADAPTER_PLUGIN_VERSION = "1.0.0"
ARTIFACT_DIR_ENV = "DAG_ML_PROCESS_ARTIFACT_DIR"
DEFAULT_ARTIFACT_DIR = "artifacts"


def emit_description() -> None:
    json.dump(
        {
            "schema_version": PROCESS_ADAPTER_DESCRIPTION_SCHEMA_VERSION,
            "protocol": PROCESS_ADAPTER_PROTOCOL,
            "adapter_id": ADAPTER_ID,
            "supported_modes": PROCESS_ADAPTER_MODES,
            "capabilities": sorted(set(PROCESS_ADAPTER_CAPABILITIES)),
        },
        sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_json(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    sys.stdout.flush()


def emit_ack(status: str) -> None:
    emit_json(
        {
            "type": "ack",
            "schema_version": PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
            "status": status,
        }
    )


def emit_error(code: str, message: str, retryable: bool = False) -> None:
    emit_json(
        {
            "type": "error",
            "schema_version": PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
            "error": {
                "code": code,
                "message": message,
                "retryable": retryable,
            },
        }
    )


# Keep the coordinator handshake cheap; runtime modes import sklearn below.
if len(sys.argv) > 1 and sys.argv[1] == "--describe":
    emit_description()
    raise SystemExit(0)


import signal
from typing import Callable, TypeVar

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

FIT_TIMEOUT_ENV = "DAG_ML_PROCESS_FIT_TIMEOUT_SECONDS"

T = TypeVar("T")


class AdapterTaskError(Exception):
    """Structured task-level error raised inside `emit_result` paths.

    In JSONL mode the loop catches it and emits an `error` frame,
    keeping the persistent worker alive for the next task. In one_shot
    mode `main` catches it and exits non-zero so the existing single
    invocation contract holds.
    """

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


def fit_timeout_seconds() -> int:
    raw = os.environ.get(FIT_TIMEOUT_ENV)
    if not raw:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


def with_fit_timeout(callable_: Callable[[], T]) -> T:
    """Run `callable_` under the controller's fit timeout, if any.

    The timeout is applied via `signal.SIGALRM` (Unix), which is the
    only portable interrupt for blocking C extensions like sklearn's
    fit loops. If `FIT_TIMEOUT_ENV` is unset or non-positive the
    helper is a no-op so smoke campaigns retain their current
    behavior.

    Timeouts surface as a retryable `fit_timeout` `AdapterTaskError`
    so the scheduler can retry on a different worker without killing
    the persistent pool.

    Note: invoking the helper cancels any pre-existing `SIGALRM`
    pending on the process. Embedding this controller inside a host
    that schedules its own SIGALRM-based deadlines is not supported.
    Stand-alone controller processes (the production deployment
    shape) do not hit this case.
    """
    seconds = fit_timeout_seconds()
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        return callable_()

    def _on_alarm(_signum: int, _frame: Any) -> None:
        raise AdapterTaskError(
            "fit_timeout",
            f"fit/predict exceeded the {seconds}s budget",
            retryable=True,
        )

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(seconds)
    try:
        return callable_()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


# Whitelisted sklearn classes. The selector keys are also used as
# stable short names in NodeTask params (e.g. `{"operator": "Ridge"}`)
# alongside the fully qualified form (`sklearn.linear_model.Ridge`).
OPERATOR_SELECTORS: dict[str, tuple[str, str]] = {
    # preprocessing
    "StandardScaler": ("sklearn.preprocessing", "StandardScaler"),
    "MinMaxScaler": ("sklearn.preprocessing", "MinMaxScaler"),
    "RobustScaler": ("sklearn.preprocessing", "RobustScaler"),
    "MaxAbsScaler": ("sklearn.preprocessing", "MaxAbsScaler"),
    "Normalizer": ("sklearn.preprocessing", "Normalizer"),
    "QuantileTransformer": ("sklearn.preprocessing", "QuantileTransformer"),
    "PowerTransformer": ("sklearn.preprocessing", "PowerTransformer"),
    # linear_model
    "LinearRegression": ("sklearn.linear_model", "LinearRegression"),
    "Ridge": ("sklearn.linear_model", "Ridge"),
    "Lasso": ("sklearn.linear_model", "Lasso"),
    "ElasticNet": ("sklearn.linear_model", "ElasticNet"),
    "LogisticRegression": ("sklearn.linear_model", "LogisticRegression"),
    "SGDRegressor": ("sklearn.linear_model", "SGDRegressor"),
    "SGDClassifier": ("sklearn.linear_model", "SGDClassifier"),
    # ensemble
    "RandomForestRegressor": ("sklearn.ensemble", "RandomForestRegressor"),
    "RandomForestClassifier": ("sklearn.ensemble", "RandomForestClassifier"),
    "GradientBoostingRegressor": ("sklearn.ensemble", "GradientBoostingRegressor"),
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "ExtraTreesRegressor": ("sklearn.ensemble", "ExtraTreesRegressor"),
    "ExtraTreesClassifier": ("sklearn.ensemble", "ExtraTreesClassifier"),
    # decomposition
    "PCA": ("sklearn.decomposition", "PCA"),
    "TruncatedSVD": ("sklearn.decomposition", "TruncatedSVD"),
    "FastICA": ("sklearn.decomposition", "FastICA"),
    "KernelPCA": ("sklearn.decomposition", "KernelPCA"),
}


def fail(message: str, code: str = "adapter_fail", retryable: bool = False) -> None:
    raise AdapterTaskError(code, message, retryable)


def resolve_operator(name: str) -> type:
    """Return the sklearn class for `name`.

    Accepts a short name (key of `OPERATOR_SELECTORS`) or a fully
    qualified `module.ClassName` string that resolves to the same
    `(module, ClassName)` pair declared in the registry. Anything
    outside the registry is rejected — the registry is the
    whitelist.
    """
    if name in OPERATOR_SELECTORS:
        module_name, class_name = OPERATOR_SELECTORS[name]
    else:
        if "." not in name:
            fail(
                f"unknown operator `{name}`; not in OPERATOR_SELECTORS registry",
                code="unknown_operator",
            )
        module_name, class_name = name.rsplit(".", 1)
        whitelisted = (module_name, class_name) in OPERATOR_SELECTORS.values()
        if not whitelisted:
            fail(
                f"operator `{name}` is not whitelisted in OPERATOR_SELECTORS",
                code="unknown_operator",
            )
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name, None)
    if klass is None:
        fail(f"operator `{name}` not found in module `{module_name}`", code="unknown_operator")
    return klass


def stable_handle(value: str) -> int:
    acc = 17
    for byte in value.encode("utf-8"):
        acc = ((acc * 31) + byte) & ((1 << 64) - 1)
    return acc


def content_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sample_scalar(sample_id: str) -> float:
    return (stable_handle(sample_id) % 10_000) / 10_000.0


_SYNTHESIS_WARNING_EMITTED = False


def warn_synthetic_fallback(task: dict[str, Any]) -> None:
    """Warn once per process when synthetic data is used despite the task
    carrying a real data_views payload.

    Slice F.1 does not wire the controller to the dag-ml-data provider
    yet, so feature/target synthesis from sample IDs is the only
    available path. A downstream operator who connects this controller
    in front of a real provider should see a clear warning before
    treating these predictions as real.
    """
    global _SYNTHESIS_WARNING_EMITTED
    if _SYNTHESIS_WARNING_EMITTED:
        return
    views = task.get("data_views") or {}
    if any(view.get("sample_ids") for view in views.values()):
        _SYNTHESIS_WARNING_EMITTED = True
        print(
            "dag-ml-sklearn-production-controller: WARNING — task carries data_views "
            "with sample_ids but features/targets are synthesized from sample IDs "
            "(provider fetch is not wired in Slice F.1).",
            file=sys.stderr,
        )


def features(sample_ids: list[str]) -> np.ndarray:
    rows = []
    for sample_id in sample_ids:
        x = sample_scalar(sample_id)
        rows.append([x, x * x, math.sin(x * math.pi), math.cos(x * math.pi)])
    return np.asarray(rows, dtype=float)


def targets(sample_ids: list[str]) -> np.ndarray:
    x = np.asarray([sample_scalar(sample_id) for sample_id in sample_ids], dtype=float)
    return 1.7 * x - 0.3 * x * x + np.sin(x * math.pi) * 0.2


def prediction_partition(phase: str) -> str:
    if phase == "FIT_CV":
        return "validation"
    if phase in {"REFIT", "PREDICT", "EXPLAIN"}:
        return "final"
    return "test"


def require_data_handles(task: dict[str, Any]) -> None:
    node_plan = task["node_plan"]
    input_handles = task.get("input_handles", {})
    data_views = task.get("data_views", {})
    for binding in node_plan.get("data_bindings", []):
        key = f"data:{binding['input_name']}"
        handle = input_handles.get(key)
        if handle is None:
            fail(f"node `{node_plan['node_id']}` did not receive data handle `{key}`")
        if handle.get("kind") not in {"data", "data_view"}:
            fail(f"node `{node_plan['node_id']}` received non-data/data-view handle `{key}`")
        view = data_views.get(key)
        if view is None:
            fail(f"node `{node_plan['node_id']}` did not receive data view spec `{key}`")
        if task.get("phase") == "FIT_CV" and task.get("fold_id") is not None:
            if view.get("partition") != "fold_train":
                fail(f"node `{node_plan['node_id']}` received non-train fold view `{key}`")
            validation_key = f"{key}:validation"
            validation_view = data_views.get(validation_key)
            if validation_view is None or validation_view.get("partition") != "fold_validation":
                fail(f"node `{node_plan['node_id']}` did not receive validation view `{validation_key}`")
        if task.get("phase") == "REFIT" and view.get("partition") != "full_train":
            fail(f"node `{node_plan['node_id']}` received non-full-train refit view `{key}`")
        if task.get("phase") == "PREDICT" and view.get("partition") != "predict":
            fail(f"node `{node_plan['node_id']}` received non-predict replay view `{key}`")


def data_view(task: dict[str, Any], suffix: str = "") -> dict[str, Any] | None:
    bindings = task["node_plan"].get("data_bindings", [])
    if not bindings:
        for view in task.get("data_views", {}).values():
            if view.get("partition") != "fold_validation":
                return view
        return next(iter(task.get("data_views", {}).values()), None)
    input_name = bindings[0]["input_name"]
    return task.get("data_views", {}).get(f"data:{input_name}{suffix}")


def train_sample_ids(task: dict[str, Any]) -> list[str]:
    view = data_view(task)
    if view is None:
        return ["sample:train:0", "sample:train:1", "sample:train:2", "sample:train:3"]
    sample_ids = view.get("sample_ids")
    if not sample_ids:
        fail(f"node `{task['node_plan']['node_id']}` train view has no sample ids")
    return list(sample_ids)


def prediction_sample_ids(task: dict[str, Any]) -> list[str]:
    phase = task["phase"]
    if phase == "FIT_CV":
        validation = data_view(task, ":validation")
        sample_ids = None if validation is None else validation.get("sample_ids")
        if not sample_ids:
            fail(f"node `{task['node_plan']['node_id']}` validation view has no sample ids")
        return list(sample_ids)
    if phase == "REFIT":
        return train_sample_ids(task)
    view = data_view(task)
    sample_ids = None if view is None else view.get("sample_ids")
    return list(sample_ids) if sample_ids else ["sample:predict:0", "sample:predict:1"]


def artifact_dir() -> str:
    return os.environ.get(ARTIFACT_DIR_ENV, DEFAULT_ARTIFACT_DIR)


def artifact_path_for(uri: str) -> str:
    """Resolve an artifact URI under the configured artifact directory.

    `joblib.load` deserializes arbitrary Python objects, so the URI is
    treated as untrusted. The basename is always taken and joined with
    `artifact_dir()` — absolute URIs and parent-directory escapes
    (`..`) are stripped. The resolved path is then asserted to live
    under `artifact_dir()` before any caller can pass it to
    `joblib.load`.
    """
    base = os.path.basename(uri)
    if not base or base in {".", ".."}:
        fail(f"refusing to resolve artifact uri `{uri}` — basename is empty or traversal")
    root = os.path.abspath(artifact_dir())
    resolved = os.path.abspath(os.path.join(root, base))
    if os.path.commonpath([root, resolved]) != root:
        fail(f"refusing to resolve artifact uri `{uri}` — outside artifact dir `{root}`")
    return resolved


def pipeline_steps_from_params(params: dict[str, Any]) -> list[tuple[str, Any]]:
    """Translate NodeTask params to a list of (name, sklearn_instance).

    Accepted shapes (the controller is strict; anything else is an
    error so dispatch bugs do not silently produce wrong models):

    1. `{"operator": "Ridge", "params": {"alpha": 1.0}}` — single step.
    2. `{"pipeline": [{"operator": "StandardScaler"},
                      {"operator": "Ridge", "params": {"alpha": 1.0}}]}`
       — multi-step pipeline; names are derived from class names
       lowercased.
    """
    if "pipeline" in params:
        spec = params["pipeline"]
        if not isinstance(spec, list) or not spec:
            fail("`pipeline` param must be a non-empty list")
        steps: list[tuple[str, Any]] = []
        for index, step in enumerate(spec):
            if not isinstance(step, dict) or "operator" not in step:
                fail(f"pipeline step {index} missing `operator`")
            klass = resolve_operator(step["operator"])
            kwargs = step.get("params") or {}
            if not isinstance(kwargs, dict):
                fail(f"pipeline step {index} `params` must be an object")
            steps.append((f"{klass.__name__.lower()}_{index}", klass(**kwargs)))
        return steps
    if "operator" in params:
        klass = resolve_operator(params["operator"])
        kwargs = params.get("params") or {}
        if not isinstance(kwargs, dict):
            fail("`params` for single operator must be an object")
        return [(klass.__name__.lower(), klass(**kwargs))]
    fail("params must define either `operator` or `pipeline`")
    return []


def make_estimator(task: dict[str, Any]) -> Pipeline:
    params = task["node_plan"].get("params") or {}
    seed = task.get("seed")
    steps = pipeline_steps_from_params(params)
    for _, step in steps:
        if seed is not None and "random_state" in getattr(step, "get_params", lambda: {})():
            step.set_params(random_state=int(seed) & 0x7FFFFFFF)
    return Pipeline(steps)


def write_artifact(estimator: Pipeline, artifact_id: str, variant_label: str) -> tuple[str, str, int]:
    target_dir = os.path.abspath(artifact_dir())
    os.makedirs(target_dir, exist_ok=True)
    fingerprint = content_fingerprint(f"{artifact_id}:{variant_label}")
    path = os.path.join(target_dir, f"{fingerprint}.joblib")
    joblib.dump(estimator, path)
    size_bytes = os.path.getsize(path)
    return path, fingerprint, size_bytes


def replay_estimator(task: dict[str, Any]) -> Pipeline:
    artifact_handles = {
        key: handle
        for key, handle in task.get("input_handles", {}).items()
        if key.startswith("artifact:")
    }
    if not artifact_handles:
        fail(f"node `{task['node_plan']['node_id']}` did not receive replay artifact handle")
    key, handle = next(iter(artifact_handles.items()))
    if task["node_plan"]["node_id"] not in key:
        fail(f"node `{task['node_plan']['node_id']}` received artifact handle for another node `{key}`")
    if handle.get("kind") not in {"model", "artifact"}:
        fail(f"node `{task['node_plan']['node_id']}` received invalid artifact handle `{key}`")
    artifact_input = task.get("artifact_inputs", {}).get(key)
    if artifact_input is None:
        fail(f"node `{task['node_plan']['node_id']}` did not receive artifact metadata `{key}`")
    if (
        artifact_input.get("node_id") != task["node_plan"]["node_id"]
        or artifact_input.get("controller_id") != task["node_plan"]["controller_id"]
    ):
        fail(f"node `{task['node_plan']['node_id']}` received mismatched artifact metadata `{key}`")
    uri = artifact_input.get("uri")
    if not uri:
        fail(f"node `{task['node_plan']['node_id']}` artifact metadata `{key}` has no uri")
    path = artifact_path_for(uri)
    if not os.path.exists(path):
        fail(
            f"node `{task['node_plan']['node_id']}` artifact uri `{uri}` "
            f"resolved under artifact dir to `{path}` which does not exist"
        )
    return joblib.load(path)


def require_prediction_inputs(task: dict[str, Any]) -> None:
    node_plan = task["node_plan"]
    input_handles = task.get("input_handles", {})
    prediction_inputs = task.get("prediction_inputs", {})
    for key, spec in prediction_inputs.items():
        handle = input_handles.get(key)
        if handle is None:
            fail(f"node `{node_plan['node_id']}` did not receive prediction handle `{key}`")
        if handle.get("kind") != "prediction":
            fail(f"node `{node_plan['node_id']}` received non-prediction handle `{key}`")
        if spec.get("producer_node") not in key:
            fail(f"node `{node_plan['node_id']}` received mismatched prediction spec `{key}`")


def require_variant_param_overrides(task: dict[str, Any]) -> None:
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    params = node_plan.get("params", {})
    variant = task.get("variant")
    if variant is None:
        return
    for dimension_name, choice in variant.get("choices", {}).items():
        choice_label = choice.get("label", "<unknown>")
        for override in choice.get("param_overrides", []):
            if override.get("node_id") != node_id:
                continue
            for key, value in override.get("params", {}).items():
                if params.get(key) != value:
                    fail(
                        f"node `{node_id}` missing generated param override "
                        f"`{dimension_name}.{choice_label}.{key}`"
                    )


def model_result(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    phase = task["phase"]
    node_id = task["node_plan"]["node_id"]
    controller_id = task["node_plan"]["controller_id"]
    variant_label = task.get("variant_id") or "base"
    fold_label = task.get("fold_id") or "nofold"

    if phase == "PREDICT":
        estimator = replay_estimator(task)
    else:
        warn_synthetic_fallback(task)
        estimator = make_estimator(task)
        ids = train_sample_ids(task)
        with_fit_timeout(lambda: estimator.fit(features(ids), targets(ids)))

    pred_ids = prediction_sample_ids(task)
    values = [
        [float(value)] for value in with_fit_timeout(lambda: estimator.predict(features(pred_ids)))
    ]
    predictions = [
        {
            "prediction_id": f"pred:{node_id}:{phase}:{variant_label}:{fold_label}",
            "producer_node": node_id,
            "partition": prediction_partition(phase),
            "fold_id": task.get("fold_id") if phase == "FIT_CV" else None,
            "sample_ids": pred_ids,
            "values": values,
            "target_names": ["y"],
        }
    ]

    artifacts: list[dict[str, Any]] = []
    artifact_handles: dict[str, Any] = {}
    if phase == "REFIT":
        artifact_id = f"artifact:{node_id}:sklearn:refit"
        uri, fingerprint, size_bytes = write_artifact(estimator, artifact_id, variant_label)
        handle_value = stable_handle(f"{artifact_id}:{variant_label}")
        artifact = {
            "id": artifact_id,
            "kind": "sklearn_pipeline",
            "controller_id": controller_id,
            "backend": "joblib",
            "uri": uri,
            "content_fingerprint": fingerprint,
            "size_bytes": size_bytes,
            "plugin": ADAPTER_PLUGIN,
            "plugin_version": ADAPTER_PLUGIN_VERSION,
        }
        artifacts.append(artifact)
        artifact_handles[artifact_id] = {
            "handle": handle_value,
            "kind": "model",
            "owner_controller": controller_id,
        }

    return predictions, artifacts, artifact_handles


def output_handles(task: dict[str, Any], handle_value: int) -> dict[str, Any]:
    node_plan = task["node_plan"]
    controller_id = node_plan["controller_id"]
    node_kind = node_plan.get("kind")
    outputs = {
        "out": {
            "handle": handle_value,
            "kind": "data",
            "owner_controller": controller_id,
        }
    }
    if node_kind in {"model", "tuner"}:
        outputs["oof"] = {
            "handle": handle_value,
            "kind": "prediction",
            "owner_controller": controller_id,
        }
    elif node_kind == "prediction_join":
        outputs["prediction"] = {
            "handle": handle_value,
            "kind": "prediction",
            "owner_controller": controller_id,
        }
    else:
        outputs["x_out"] = {
            "handle": handle_value,
            "kind": "data",
            "owner_controller": controller_id,
        }
    return outputs


def build_result(task: dict[str, Any]) -> dict[str, Any]:
    node_plan = task["node_plan"]
    node_id = node_plan["node_id"]
    phase = task["phase"]
    controller_id = node_plan["controller_id"]
    variant_id = task.get("variant_id")
    fold_id = task.get("fold_id")
    variant_label = variant_id or "base"
    fold_label = fold_id or "nofold"
    handle_value = stable_handle(f"{node_id}:{phase}:{variant_label}:{fold_label}")

    predictions: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    artifact_handles: dict[str, Any] = {}
    if node_plan.get("kind") in {"model", "tuner"}:
        predictions, artifacts, artifact_handles = model_result(task)

    metrics = {"sklearn_production_adapter": 1.0}
    if predictions:
        flat = [row[0] for row in predictions[0]["values"]]
        metrics["prediction_mean"] = float(sum(flat) / len(flat))
    worker_index = os.environ.get("DAG_ML_PROCESS_WORKER_INDEX")
    worker_count = os.environ.get("DAG_ML_PROCESS_WORKER_COUNT")
    if worker_index is not None:
        metrics["process_worker_index"] = float(worker_index)
    if worker_count is not None:
        metrics["process_worker_count"] = float(worker_count)

    return {
        "node_id": node_id,
        "outputs": output_handles(task, handle_value),
        "predictions": predictions,
        "shape_deltas": [],
        "artifacts": artifacts,
        "artifact_handles": artifact_handles,
        "lineage": {
            "record_id": f"lineage:{node_id}:{phase}:{variant_label}:{fold_label}",
            "run_id": task["run_id"],
            "node_id": node_id,
            "phase": phase,
            "controller_id": controller_id,
            "controller_version": node_plan["controller_version"],
            "variant_id": variant_id,
            "fold_id": fold_id,
            "branch_path": task.get("branch_path", []),
            "input_lineage": [],
            "artifact_refs": artifacts,
            "params_fingerprint": node_plan["params_fingerprint"],
            "data_model_shape_fingerprint": None,
            "aggregation_policy_fingerprint": None,
            "seed": task.get("seed"),
            "unsafe_flags": [],
            "metrics": metrics,
        },
    }


def emit_result(task: dict[str, Any]) -> None:
    require_data_handles(task)
    require_prediction_inputs(task)
    require_variant_param_overrides(task)
    emit_json(build_result(task))


def emit_result_frame(task: dict[str, Any]) -> None:
    require_data_handles(task)
    require_prediction_inputs(task)
    require_variant_param_overrides(task)
    emit_json(
        {
            "type": "result",
            "schema_version": PROCESS_ADAPTER_FRAME_SCHEMA_VERSION,
            "result": build_result(task),
        }
    )


def is_control_frame(payload: Any) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("type"), str)


def validate_frame_schema(frame: dict[str, Any]) -> bool:
    if frame.get("schema_version") != PROCESS_ADAPTER_FRAME_SCHEMA_VERSION:
        emit_error(
            "unsupported_frame_schema",
            f"unsupported frame schema version `{frame.get('schema_version')}`",
            retryable=False,
        )
        return False
    return True


def handle_control_frame(frame: dict[str, Any]) -> bool:
    if not validate_frame_schema(frame):
        return True
    frame_type = frame["type"]
    if frame_type == "init":
        write_lifecycle_marker("init", frame)
        emit_ack("initialized")
        return True
    if frame_type == "task":
        task = frame.get("task")
        if not isinstance(task, dict):
            emit_error("invalid_task_frame", "task frame is missing object field `task`")
            return True
        emit_result_frame(task)
        return True
    if frame_type == "close":
        write_lifecycle_marker("close", frame)
        emit_ack("closed")
        return False
    emit_error("unsupported_frame", f"unsupported frame type `{frame_type}`")
    return True


def write_lifecycle_marker(event: str, frame: dict[str, Any]) -> None:
    marker_dir = os.environ.get("DAG_ML_PROCESS_LIFECYCLE_MARKER_DIR")
    if not marker_dir:
        return
    os.makedirs(marker_dir, exist_ok=True)
    controller_id = frame.get("controller_id") or os.environ.get("DAG_ML_CONTROLLER_ID", "controller")
    worker_index = (
        frame.get("worker_index")
        if frame.get("worker_index") is not None
        else os.environ.get("DAG_ML_PROCESS_WORKER_INDEX", "0")
    )
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in f"{event}_{controller_id}_{worker_index}"
    )
    with open(os.path.join(marker_dir, f"{safe_name}.marker"), "a", encoding="utf-8") as marker:
        marker.write(event)
        marker.write("\n")


def run_jsonl() -> None:
    """Persistent-worker loop.

    Task-level errors surface as structured `error` frames so the
    worker survives the bad task and processes the next one. The loop
    only terminates on `close` (clean shutdown) or on EOF on stdin.
    """
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            emit_error("invalid_task_json", f"invalid NodeTask JSON line: {exc}", retryable=False)
            continue
        try:
            if is_control_frame(payload):
                if not handle_control_frame(payload):
                    break
                continue
            emit_result(payload)
        except AdapterTaskError as exc:
            emit_error(exc.code, exc.message, exc.retryable)
        except Exception as exc:  # noqa: BLE001 — surface as a structured frame, not a crash
            emit_error(
                "adapter_unexpected_error",
                f"{type(exc).__name__}: {exc}",
                retryable=False,
            )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--describe":
        emit_description()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--jsonl":
        run_jsonl()
        return
    try:
        task = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(f"invalid NodeTask JSON: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    try:
        emit_result(task)
    except AdapterTaskError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
