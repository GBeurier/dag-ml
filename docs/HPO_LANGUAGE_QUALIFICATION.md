# Host HPO and portable Raw qualification

The native DAG-ML scheduler owns folds, prediction scoring, trial concurrency,
pruning, selection and durable checkpoint recovery. Language adapters provide
candidate parameters and execute operators; their presence alone is not proof
that a local model fits on the declared training fold.

| Surface | Executable evidence | Current boundary |
| --- | --- | --- |
| C ABI | Parallel candidate callbacks and checkpoint recovery in `dag-ml-capi` tests | A caller must supply its own operator and optimizer callbacks. |
| Python/CLI | Public nirs4all RandomForest and other model HPO oracles; CLI process tests | Python object artifacts remain Python-host. |
| Node WASM and Chrome Web Workers | Real scalar Ridge fitted on each fold's train IDs; per-fold and pooled OOF RMSE, parallel work, pruning and resume | The example does not imply all JavaScript operator families are implemented. |
| R/CLI | `r_hpo_ridge` fits Ridge in R using the signed plan FoldSet and an independent numeric table, through `dagml_host_hpo_search()` and the CLI; it checks native scores, parallel trials, pruning, selection, resume, REFIT, fresh-process RDS replay and rejection of a corrupted sidecar | This qualifies one R-hosted operator, not every R model or a portable RDS artifact. |
| Octave/CLI | `octave_hpo_ridge` fits Ridge in Octave from the signed plan FoldSet and a numeric table; CI checks fold evidence, native parallel HPO, pruning, resume, REFIT, fresh-process MAT replay and rejection of a corrupted sidecar | This qualifies one Octave-hosted operator. Licensed MATLAB execution is outside the active qualification scope. |

The optional Python/CLI N4M adapter `examples/adapters/hpo_n4m_optimizer.sh`
replaces the example's trial-index proposals with a persisted Methods
optimizer. With `DAGML_N4M_PYTHON` pointing to a local n4m installation, the
CLI integration test exercises native fold scores, Sobol continuation,
Median pruning and two recovery windows. It uses the same simple DAG fixture
as the stateless CLI adapters; comparison against the public nirs4all
multimodal DAG, installed wheels and other language hosts remains separate.

Run the strict R oracle with `Rscript` and `jsonlite` installed:

```sh
DAGML_REQUIRE_HPO_R=1 cargo test -p dag-ml-cli --test r_hpo_ridge
```

Run the Octave oracle without a MATLAB licence:

```sh
DAGML_REQUIRE_HPO_OCTAVE=1 cargo test -p dag-ml-cli --test octave_hpo_ridge
```

The core and bindings can carry signed Raw artifact bytes and ask the host to
export, hydrate and release them. This is a transport contract: replay of a
particular model across languages additionally needs a versioned fitted-state
codec and a matching inference controller in each language. Methods PLS with
N4MM is an operator-specific portable example. RDS, MATLAB objects and Python
joblib cannot be relabelled as portable Raw merely because their bytes fit the
transport.
