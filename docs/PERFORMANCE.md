# Performance Probes

`dag-ml` does not treat microbenchmarks as public API. The 0.2.x RC
baseline is a set of ignored Rust tests that can be run on demand in release
mode to catch large regressions on critical control paths.

## Current Probes

| Probe | Command | Purpose |
|---|---|---|
| OOF campaign join | `cargo test -p dag-ml-core oof_join_large_campaign_under_1500ms --release -- --ignored --nocapture` | Joins 12k samples across 4 producers and 6 folds by stable sample id. |
| Execution-plan build | `cargo test -p dag-ml-core build_execution_plan_large_linear_graph_under_1500ms --release -- --ignored --nocapture` | Builds a 401-node linear graph into a validated `ExecutionPlan`. |

## Policy

- Probes are intentionally private tests, not exported signatures.
- They should be run before releases and after scheduler, OOF, graph
  or planning rewrites.
- Thresholds are generous sanity gates, not formal service-level objectives.
- If a probe becomes flaky on CI hardware, keep the measurement but move the
  threshold to a manual release checklist rather than weakening correctness
  tests.

## Next Baselines

Post-0.2.x, add probes for:

- prediction-cache row-major and columnar export;
- bundle replay data/artifact/prediction-cache materialization;
- process-adapter persistent worker pool throughput on the branch/merge smoke.
