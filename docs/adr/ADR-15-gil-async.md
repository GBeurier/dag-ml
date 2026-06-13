# ADR-15: Python GIL / async policy

**Status**: accepted (2026-05-29)
**Blocks**: workstream D (Python packaging), workstream E (bridge)

## Context

The PyO3 bindings (workstream D) expose `dag-ml` and `dag-ml-data` to Python. Long-running Rust calls (FIT_CV, REFIT, PREDICT) must not block other Python threads holding the GIL, and host controllers re-entering Python from Rust worker threads must do so safely. nirs4all uses `joblib.Parallel` and sklearn/BLAS thread pools; uncoordinated nesting oversubscribes the CPU (Codex hidden risk).

## Decision

1. **Release the GIL on long-running calls** — every binding entry point that runs a phase (`fit_cv`, `select`, `refit`, `predict`, `explain`) or builds/validates a bundle wraps the Rust call in `py.allow_threads(...)`. Short metadata/validation calls keep the GIL (the acquire/release overhead would dominate).
2. **Controller re-entry contract** — a host controller invoked from a Rust worker thread re-acquires the GIL (`Python::with_gil`) before touching Python objects. The contract is documented for controller authors: *your `invoke` may run on a non-main thread; acquire the GIL yourself.*
3. **No asyncio in v1** — `nirs4all.run / predict / explain / retrain` stay synchronous. An async facade is explicitly out of scope (descoped in the roadmap). Concurrency is achieved by GIL-released Rust scheduling, not by Python coroutines.
4. **Thread-pool ceiling** — `nirs4all.run(n_jobs=N)` maps to the dag-ml scheduler worker count. The docs instruct operators to pin BLAS/OpenMP pools (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`) and to avoid stacking `joblib.Parallel(prefer="threads")` on top of an already-parallel scheduler. The bridge logs the effective worker count and detected BLAS thread count at startup so oversubscription is visible.

## Consequences

- Workstream D wraps phase calls in `allow_threads`; the controller ABI doc (ADR-13 worker-process context) states the GIL re-entry contract.
- The observability spans (ADR-12) carry the worker-thread id so cross-thread controller invocations are traceable.
- nirs4all's existing synchronous API surface is preserved exactly.

## Risk

- Releasing the GIL exposes any unsafe shared state in host controllers. The contract is documented and the default sklearn/R adapters are process-isolated (ADR-13), so the GIL-release path only touches in-process PyO3 controllers, which are the advanced case. In-process controllers that are not thread-safe must declare `thread_safe = false` in their manifest; the scheduler then serializes them.
