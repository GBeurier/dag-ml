# Examples

These fixtures are part of the public integration surface. Keep them small,
deterministic, and runnable from the commands documented in `docs/TEST_PLAN.md`.

| Audience | Start here | Purpose |
|---|---|---|
| Pipeline author | `pipeline_dsl_nirs4all_compat.json` | Compile the nirs4all-compatible JSON profile without importing nirs4all. |
| Host-controller author | `adapters/python_process_controller.py` and `controller_manifests.json` | Exercise process-controller handshakes, manifests, task input, and result validation. |
| Runtime integrator | `branch_merge_oof_graph.json`, `campaign_branch_merge_oof.json`, `fixtures/bundle/` | Validate FIT_CV, OOF stacking, REFIT bundle capture, and replay contracts. |
| Data-contract integrator | `fixtures/data/coordinator_data_plan_envelope_nir.json` | Use the shared `dag-ml-data` envelope fixture pinned by `scripts/validate_contracts.py`. |
| Provenance consumer | `generated/README.md` | Inspect generated bundle, prediction-cache, artifact-manifest, and research-provenance outputs. |

Examples that become compatibility evidence must be added to the conformance
pack or parity oracle before release.
