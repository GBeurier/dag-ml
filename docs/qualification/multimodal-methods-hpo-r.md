# R Methods nested HPO and complete predictor archives

`scripts/qualify_multimodal_methods_hpo_r.py` qualifies the installed public
`nirs4all` R Methods controller with the numeric inputs and optimizer proposals
from a fresh `scripts/smoke_wasm_multimodal_methods_hpo.mjs` capture. It does not
generate data, implement numerical kernels or run a host CV loop.

The campaign first invokes `dag-ml-cli run-host-hpo` with the actual persistent
R executable adapter. Rust schedules three recorded proposals across the same
outer group folds and nested inner OOF folds as the Node campaign. Each
candidate receives an isolated R worker. A Python binding invocation of the
same native search independently uses the original physical row order; the
CLI invocation uses a different physical row permutation for each named source
and for the target table. Scores and the selected proposal must match the
fresh Node result within the explicitly recorded tolerance.

Native CV, SELECT and REFIT then capture four source estimators and the
OOF-trained meta estimator. All five are actual `n4m` role models, with native
N4ME state in bounded RAW wrappers. The Python training result exports them
with explicit `fitted_artifact_mode="portable_required"` and
`artifact_load_mode="native_portable"`; its general host-sidecar default is
not selected for this package. The public SDK writes their unchanged
package through Core Archive V2 to `.n4a`, reopens it with Core and DAG semantic
validation, and replays it through a fresh installed R adapter process. Replay
receives only named current heldout sources, no target values and
`allow_fit=FALSE`. Its lifecycle must show five hydrations, five predictions,
five releases and no fits, including native disposal evidence. The final
heldout prediction and native score reports are compared with the Node proof.

The campaign also refuses unsigned phase changes, mismatched current trust,
altered or missing ZIP members, member aliases, a same-width feature-name
permutation, and an explicit fit call on a replay adapter. An injected
prediction failure must release all five hydrated states. These checks retain
package bindings and native recipe validation; they do not deserialize Python
models or convert the five role estimators to a single N4MM model.

## Run after complete implementation review

Use fresh matching DAG, Core, SDK and R installations. The R library must
contain the reviewed `nirs4all` package plus the matching `n4m` and `dagml`
packages. Provide the fresh Node capture and CLI as explicit files. The
qualification work directory must not already exist.

```sh
export R_LIBS_USER=/absolute/path/to/isolated/rlib
export PATH=/home/delete/miniconda3/bin:$PATH
python scripts/qualify_multimodal_methods_hpo_r.py \
  --node-capture /absolute/path/to/fresh-node-capture.json \
  --cli /absolute/path/to/dag-ml-cli \
  --workdir /absolute/path/to/new-r-qualification
```

The directory retains the exact CLI command, input contracts, recorded
proposals, actual R lifecycle logs, package and outcome, `.n4a`, and
`qualification-receipt.json`. The receipt records Python facade/native module
paths, installed R package paths/versions and loaded DLL hashes. A receipt is
written only after every positive and negative qualification succeeds.

The worker-failure lifecycle test runs independently of numerical libraries.
It uses real child processes that return an invalid close acknowledgement or
stop replying while ignoring EOF and SIGTERM. Both must have their pipes
closed and be killed and reaped, while retaining the original failure. A
controller error inside a worker context must also survive shutdown failures.

```sh
python -m unittest discover -s tests \
  -p test_multimodal_methods_hpo_r_campaign.py -k WorkerFailureLifecycleTests -v
```

The independent integration entry point can require the real external runtime
instead of skipping when it is absent:

```sh
export NIRS4ALL_R_ROLE_NODE_CAPTURE=/absolute/path/to/fresh-node-capture.json
export DAG_ML_CLI=/absolute/path/to/dag-ml-cli
export NIRS4ALL_REQUIRE_R_ROLE_CAMPAIGN=1
python -m pytest tests/test_multimodal_methods_hpo_r_campaign.py -q
```

## Public R host search

Prepare current named source matrices and explicit sample IDs independently
from the native signed envelope and plan. Supply the current compiled operator
map and an independently trusted current R controller manifest. The adapter
constructor validates the source and target tables; native DAG-ML owns the
plan, folds, RNG, metrics, selection and checkpoints.

```r
prepared <- nirs4all::nirs4all_dag_role_adapter(
  sources = sources, operators = compiled_operators,
  controller_manifest = current_r_manifest,
  trusted_manifest = independently_trusted_r_manifest,
  targets = list(values = y, sample_ids = target_sample_ids),
  target_names = "y", allow_fit = TRUE,
  workdir = tempfile("methods-r-hpo-")
)
search <- dagml::dagml_host_hpo_search(
  plan = "plan.json", envelope = "envelope.json", request = "hpo-request.json",
  operator_adapter = prepared$adapter,
  optimizer_adapter = "/absolute/path/to/optimizer-jsonl-executable",
  cli = "/absolute/path/to/dag-ml-cli",
  operator_persistent = TRUE, parallel_trials = 1L,
  adapter_timeout_ms = 120000L,
  output = "hpo-result.json"
)
```

The recorded optimizer executable emitted by this qualification is a fixture
for reproducing the Node proposals, not an optimizer implementation or a data
generator. Product applications supply their actual public optimizer adapter.

This proof covers four numeric projections (NIR, image, series and metadata)
and an OOF meta Ridge. It does not establish raw N-D, MATLAB/Octave or any
additional scientific encoding qualification. The fixture's native-derived
training influence remains immutable because its node, fold and relation
coordinates are unchanged; Rust validates it against the current R training
projection before callbacks.
