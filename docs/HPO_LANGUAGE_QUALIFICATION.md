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
| R/CLI | `r_hpo_ridge` fits Ridge in R using the signed plan FoldSet and an independent two-row numeric table, through `dagml_host_hpo_search()` and the CLI; it checks native scores, parallel trials, pruning, selection and resume | This qualifies the R host HPO path for one operator. REFIT and artifact replay require a separate operator and execution oracle. |
| Octave/MATLAB/CLI | CI installs Octave and checks the wrapper, optimizer JSONL protocol and native scheduler with a synthetic Python operator | A real Octave/MATLAB operator with numeric fold evidence is still missing. MATLAB itself is not run in CI. |

Run the strict R oracle with `Rscript` and `jsonlite` installed:

```sh
DAGML_REQUIRE_HPO_R=1 cargo test -p dag-ml-cli --test r_hpo_ridge
```

The core and bindings can carry signed Raw artifact bytes and ask the host to
export, hydrate and release them. This is a transport contract: replay of a
particular model across languages additionally needs a versioned fitted-state
codec and a matching inference controller in each language. Methods PLS with
N4MM is an operator-specific portable example. RDS, MATLAB objects and Python
joblib cannot be relabelled as portable Raw merely because their bytes fit the
transport.
