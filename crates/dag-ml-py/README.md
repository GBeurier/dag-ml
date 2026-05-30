# dag-ml Python bindings

Thin PyO3/maturin bindings for DAG-ML JSON contracts.

This package validates, compiles and plans serialized DAG-ML contracts. It does
not execute host controllers, own fitted model objects or materialize data
buffers.

## Build

```bash
cargo test -p dag-ml-py
maturin build --release --features extension-module
python3 ../../scripts/smoke_python_bindings.py  # after installing the wheel
```

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
```

All Rust-side validation failures are raised as `dag_ml.DagMlError`. Native
errors expose `category`, `code`, `severity`, `remediation_hint`, `context`,
`context_json` and `descriptor_json` attributes for ADR-11-compatible handling.
