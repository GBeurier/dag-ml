# Host HPO trial concurrency: proposed native contract

Status: bounded concurrent execution is implemented for non-durable trials
without progressive pruning. `n_jobs > 1` asks a window of proposals in trial
order, evaluates them on separate native workers, and tells the optimizer in
trial order. The Python binding releases the GIL while those workers run and
creates a separate data provider, operator callback, resolver and artifact
store for each candidate. The native test and PyO3 test assert actual overlap.
Durable storage/resume and progressive pruning remain unsupported for parallel
trials until the checkpoint and intermediate-feedback parts below are added.

Binding availability is narrower than the Rust core contract. Rust hosts can
call `SequentialScheduler::execute_parallel_host_hpo_search_with_candidate_factories`
with their own proposal source and candidate-local controller/provider factories;
the core test exercises this public method. The PyO3 binding implements those
factories and has a direct overlap test. The `dag-ml-cli` has no host HPO command,
and the C ABI does not expose host HPO proposal or factory callbacks, so R,
MATLAB and WASM bindings cannot yet invoke this search. A Python nirs4all outer
run may use the CLI for its ordinary pipeline while still invoking PyO3 for
the nested host HPO. That is not CLI HPO support.

The current core loop is `ask -> FIT_CV -> tell -> checkpoint`. The Python
binding supplies one `InMemoryDataProvider`, whose handle maps use `RefCell`,
and nirs4all's operator callback uses a shared mutable resolver and artifact
store. Running this loop on several threads would share candidate-local state
and allow one trial's handles or fitted model to enter another trial.

The remaining durable/pruning extension builds on a coordinator-owned window
of at most `max_parallel_trials` candidates:

1. Implemented: the coordinator calls the proposal source's `ask` on one
   thread, assigns a stable trial index and immutable parameter overrides,
   then dispatches each candidate to a worker. No worker calls the mutable
   proposal source.
2. Implemented for the Python host: each worker receives its own data-provider
   instance, `RunContext`, handle namespace, operator callback, resolver and
   artifact namespace. Other language hosts must provide equivalent factories.
3. Workers return native fold reports and candidate evidence. For progressive
   pruning, a worker sends an intermediate score to the coordinator and waits
   for its prune/continue decision. The coordinator alone calls the host
   optimizer's `report_intermediate`, `pruned`, `tell` or `fail` callback.
4. Completed trials may arrive out of order. The durable checkpoint must
   record each trial's stable index and terminal state without pretending the
   terminal list is already contiguous; active proposals must be represented
   or a resume must fail closed. A phase's next sampler starts only after all
   prior-phase trials are terminal.

The binding and host must pair the native checkpoint with the optimizer study
after each terminal transition. Tests should prove overlapping candidate
execution, per-trial data/artifact isolation, truthful pruning, a failed trial
beside a successful one, CLI/PyO3 parity, and restart with out-of-order
terminal evidence. Equal scores for a fixed candidate are required; identical
sampling order to sequential Optuna is not, because `n_jobs > 1` asks while
other trials are still running.
