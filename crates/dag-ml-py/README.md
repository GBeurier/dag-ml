# dag-ml Python bindings

Thin PyO3/maturin bindings for DAG-ML JSON contracts.

This package validates, compiles and plans serialized DAG-ML contracts. Its
owning training entry point also executes the native DAG-ML coordinator while
operator implementations remain Python callbacks; no numerical or fold logic
is reimplemented in the binding.

## Build

This crate is excluded from the root cargo workspace (its `abi3-py311` floor
would force a Python >= 3.11 host on `cargo test --workspace` / `cargo
llvm-cov`), so build and test it through its own manifest against a Python
>= 3.11 interpreter:

```bash
PYO3_PYTHON=python3.11 cargo test --manifest-path crates/dag-ml-py/Cargo.toml
maturin build --release --features extension-module   # run from this crate dir
python3 ../../scripts/smoke_python_bindings.py        # after installing the wheel
PYTHONPATH=python python3.11 -m unittest discover -s tests
```

The source package also contains the tracked `_dag_ml.abi3.so` used by direct
`PYTHONPATH=crates/dag-ml-py/python` imports. After changing compiled Rust
inputs, refresh it with `maturin develop --release` inside an active Python
3.11+ virtual environment, run `python scripts/check_so_freshness.py`, and
smoke the public source-tree import. The freshness gate rejects dirty or
untracked Rust inputs when that tracked extension is unchanged.

## Python Surface

```python
import dag_ml

dag_ml.validate_graph_json(graph_json)  # raw JSON helper remains available

dsl = dag_ml.PipelineDslSpec(dsl_json)
controllers = dag_ml.ControllerManifests(controller_manifests_json)
artifact = dag_ml.compile_pipeline_dsl_artifact(dsl)
plan = dag_ml.build_execution_plan(
    "plan:example",
    artifact.graph,
    artifact.campaign_template,
    controllers,
)
plan_json = plan.json()

validated_request = dag_ml.TrainingRequest.from_path(
    "examples/fixtures/training/training_request_active_influence.v1.json"
)
validated_request = dag_ml.sign_training_request(unsigned_training_request)
relation_fingerprint = dag_ml.sample_relation_set_fingerprint_json(relations_json)
training_projection = validated_request.project()
package = dag_ml.PortablePredictorPackage.from_path(
    "examples/fixtures/training/portable_predictor_package.v1.json"
)

result = dag_ml.execute_training(
    native_training_request,
    data_envelopes={"model:base.x": signed_envelope},
    relations=sample_relations,
    training_influence=signed_influence,
    op_callback=run_node,
    outcome_id="outcome:example",
    run_id="run:example",
    bundle_id="bundle:example",
)
bundle = result.execution_bundle
scores = result.score_set
portable_artifacts = result.artifacts
result.detach()  # explicitly release callbacks, views and artifact handles
```

Portable package replay is available without the original `TrainingResult`.
Hosts pass the signed package, a `TrainingReplayRequest` whose `phase` is either
`PREDICT` or `EXPLAIN`, the current cohort data envelopes, and explicit sidecar
artifact handles:

```python
outcome = dag_ml.replay_loaded_predictor_package(
    package,
    replay_request,  # {"phase": "PREDICT"} or {"phase": "EXPLAIN"}
    data_envelopes,
    artifact_handles,
    run_node,
    outcome_id="outcome:package.replay",
    run_id="run:package.replay",
)
```

`PREDICT` must reproduce the requested package output bindings exactly.
`EXPLAIN` must emit at least one explanation block and may include the final
bound predictions for the requested bindings. The package remains handle-free;
all process-local model handles are supplied through `artifact_handles`.

`TrainingRequest`, `TrainingContractProjection`, `ParameterProjection`,
`CacheNamespace` and `PortablePredictorPackage` are validated by the native
`dag-ml-core` contracts. `sign_training_request()` canonicalizes and signs an
unsigned request through the same native structs, and
`sample_relation_set_fingerprint_json()` exposes the core relation fingerprint
for host-side data envelope assembly. `project_training_request()` is also
available as a functional facade. The binding does not reproduce parameter projection,
capability-derived influence or portability rules in Python.

`execute_training()` requires an envelope map keyed by the exact
`node_id.input_name` requirement key plus the matching relations and influence
manifest. The PyO3 layer releases the GIL while the core runs and reacquires it
only for controller callbacks. `TrainingResult` retains the controller registry,
attested provider and artifact store until `detach()` or object destruction;
portable outcome, bundle, scores, outputs and artifact metadata remain readable
after detach, while process-local handles are never serialized.

All Rust-side validation failures are raised as `dag_ml.DagMlError`. Native
errors expose `category`, `code`, `severity`, `remediation_hint`, `context`,
`context_json` and `descriptor_json` attributes for ADR-11-compatible handling.
