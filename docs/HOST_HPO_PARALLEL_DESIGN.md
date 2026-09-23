# Host HPO concurrency and recovery

DAG-ML core owns the trial window, FIT_CV work, native fold scores, pruning feedback,
selection, and checkpoint validation. The host optimizer only proposes parameters
and receives terminal or intermediate scores. `max_parallel_trials` bounds real
concurrent candidate evaluation. The coordinator asks in trial order, workers
use separate providers, controllers and run contexts, and the coordinator tells
in trial order. A phase boundary is never crossed by one window. The host may
sample differently than a sequential optimizer, but an evaluated candidate's
score has the same native meaning.

When progressive pruning is enabled, each worker sends its native fold score
to the coordinator and waits for a prune decision. Workers never call the
optimizer. A cancelled search finishes all candidates already in flight before
publishing the window's terminal checkpoint. A failed worker terminalizes its
siblings before the first error propagates.

For durable storage, core seals a prospective checkpoint and calls
`HostHpoProgress::prepare_terminal` before the optimizer's `tell`, `pruned`, or
`fail` transition. It publishes the actual checkpoint after that transition.
The nirs4all Optuna host stores both records in the study. On restart it
validates the prepared native record, completes any interrupted optimizer
transition, and marks other in-flight candidates failed without inventing a
score. A direct fault-injection test interrupts between `tell` and checkpoint
publication, then resumes successfully. A clean restart can increase the total
trial budget without repeating historical fits.

Rust hosts use `execute_parallel_host_hpo_search_with_candidate_factories` or
its resumable counterpart. PyO3 uses the same core methods with candidate-local
operator callbacks; nirs4all has legacy/DAG PyO3 and outer CLI oracles for
parallel Optuna storage and pruning. C ABI v1 exposes non-durable parallel work;
`dagml_host_hpo_search_json_v2` exposes fold feedback, pruning and durable
checkpoint callbacks, while `dagml_host_hpo_checkpoint_recover_json` validates
recovery. Its C/Rust test proves these transitions. WASM now exposes a
single-worker `host_hpo_search_json` adapter over the same core search and
synchronous JavaScript operator/optimizer callbacks. Browser worker-parallel
HPO and idiomatic R/MATLAB wrappers remain open; those native hosts can use the
C ABI v2 or standalone CLI process protocol meanwhile.

`dag-ml-cli run-host-hpo` is a standalone host HPO command. It reads an
`ExecutionPlan`, an `ExternalDataPlanEnvelope`, and a `HostHpoSearchRequest`
from JSON files. `--operator-adapter` uses the existing process-controller
protocol; `--operator-persistent` gives each candidate its own persistent
controller process when the operator needs state across folds. An optimizer
JSONL process receives one object per line and must reply with one object per
line:

| Operation | Required reply |
| --- | --- |
| `init` | `{"prepared_checkpoint": null, "interrupted": []}` or a prepared native checkpoint plus interrupted trial proposals |
| `ask` | `{"params": {"parameter": value}}` or `{"params": null}` |
| `report_intermediate` | `{"prune": false}` or `true` |
| `tell`, `pruned`, `fail`, `prepare_terminal`, `checkpoint` | `{"ok": true}` |

`--parallel-trials` sets the worker bound. `--checkpoint PATH` enables durable
native checkpoints, written through a temporary file and rename after each
terminal transition; the optimizer adapter must persist its own paired state.
On `init`, its `prepared_checkpoint` and `interrupted` proposals are validated
and recovered by core before resuming. `--output PATH` writes the search result
as JSON. `examples/adapters/hpo_optimizer_jsonl.py` and
`examples/adapters/hpo_process_controller.py` provide a small working pair.
The CLI test runs two concurrent trials with fold feedback, then resumes to a
third trial from the native checkpoint.
