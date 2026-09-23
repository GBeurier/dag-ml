# Host HPO trial concurrency: proposed native contract

Status: design only. `HostHpoSearchRequest` currently executes one trial at a
time. A host `n_jobs > 1` must remain unsupported until the contract below is
implemented; accepting the flag while evaluating trials serially would not
match Optuna's legacy behavior.

The current core loop is `ask -> FIT_CV -> tell -> checkpoint`. The Python
binding supplies one `InMemoryDataProvider`, whose handle maps use `RefCell`,
and nirs4all's operator callback uses a shared mutable resolver and artifact
store. Running this loop on several threads would share candidate-local state
and allow one trial's handles or fitted model to enter another trial.

The smallest safe extension is a coordinator-owned window of at most
`max_parallel_trials` candidates:

1. The coordinator calls the proposal source's `ask` on one thread, assigns a
   stable trial index and immutable parameter overrides, then dispatches each
   candidate to a worker. No worker calls the mutable proposal source.
2. A worker receives its own data-provider instance, `RunContext`, handle
   namespace and host artifact namespace. The binding must create one provider
   from the attested envelope per worker and key host-side caches by trial/run
   identity. Controller implementations must attest concurrency support.
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
