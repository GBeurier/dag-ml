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
recovery. Its C/Rust test proves these transitions. WASM exposes both the
single-worker `host_hpo_search_json` adapter and the asynchronous
`host_hpo_search_parallel_json` coordinator. The latter dispatches a native,
phase-bounded candidate window to separate Web Worker/WASM instances and
reconciles their scores in trial order. R and MATLAB/Octave can supply the optimizer through
the standalone CLI JSONL process protocol (examples below), or use the C ABI
v2 from a native host; these examples are not an idiomatic language binding.

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
as JSON. `examples/adapters/hpo_optimizer_jsonl.py`,
`examples/adapters/hpo_optimizer_jsonl.R`, and
`examples/adapters/hpo_optimizer_matlab.sh` (backed by
`hpo_optimizer_jsonl.m`) implement the same deterministic optimizer protocol.
Each can be passed as `--optimizer-adapter`; the R script requires Rscript and
jsonlite, and the shell wrapper selects Octave or MATLAB. They are deliberately
stateless examples: a real adaptive optimizer must persist its own proposal
and prepared-terminal state before acknowledging checkpoint transitions.
`python3 scripts/test_hpo_language_adapters.py` checks their exact JSONL
replies, skipping only absent language runtimes. The CLI test runs two concurrent
trials with the Python adapter and fold feedback, then resumes to a third trial
from the native checkpoint.

`host_hpo_search_parallel_json` accepts a dispatcher
`(taskJson) => Promise<workerResultJson>` and a synchronous optimizer callback
with the same operations as `host_hpo_search_json`. It dispatches all tasks in
each phase-bounded window before awaiting a result. Each worker calls
`host_hpo_evaluate_worker_task_json` with its own controller callback, trusted
manifest and data envelope; this runs FIT_CV and produces native fold scores.
For `progressive_pruning`, the dispatcher receives tagged `fold` and
`complete` messages. Each `fold` evaluates only the requested FIT_CV fold and
returns a native score set; the coordinator reports its validated aggregate
to the optimizer before dispatching that candidate's next fold. A pruned
candidate has no later fold task or fabricated OOF score. Survivors run one
full native FIT_CV pass to construct the exact global OOF result and cross-check
its fold scores against all earlier feedback. This final pass adds compute for
survivors until a stateful worker context protocol is available.
Core checks the checkpoint/data fingerprint, exact candidate plan, fold
reports, derived objective and complete trial-key coverage before any optimizer
transition. `prepare_terminal` precedes ordered `tell`/`pruned`/`fail`; `checkpoint`
follows. A rejected worker Promise becomes a failed candidate with no
manufactured score. Existing sequential behavior is unchanged.

The browser host must give each candidate a fresh controller/model handle
namespace, even when reusing a Web Worker. The Node smoke uses two
`worker_threads` with separate WASM instances and checks simultaneous
dispatch, ordered terminalization, checkpoint resume, and an actual prune
before the next fold. The synchronous single-worker API also supports fold
pruning. R/MATLAB JSONL adapters remain usable as separate CLI processes.
The R and Octave CI jobs require their JSONL adapter, CLI wrapper, and native
two-worker HPO/resume tests to pass; the latter exercises actual optimizer
processes, not only a fake CLI. The web target is also exercised in headless
Chrome with two actual module `Web Worker` instances, each loading its own
WASM instance; the browser test verifies parallel dispatch and checkpoint
resume. The Node smoke additionally verifies progressive pruning.
