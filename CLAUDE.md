# DAG-ML Development Context

You are implementing DAG-ML, a Rust low-level coordinator for reproducible,
traceable and OOF/leakage-safe ML/DL/bioinformatics pipelines.

## Product Direction

- Keep operators external. The Rust core owns graph compilation, scheduling,
  replay, lineage, OOF safety, leakage validation, fingerprints and handle
  lifecycle.
- Bindings/controllers own model fitting, transforms, augmentations, data
  backends and native library integrations.
- Keep `dag-ml` and `dag-ml-data` responsibilities separate. Cross-repo
  compatibility is enforced by shared contracts and fixtures.
- Preserve DAG-ML-specific invariants internally. Research standards such as
  W3C PROV, Workflow Run RO-Crate, OpenLineage and MLMD are export targets,
  not the internal execution model.

## Current Priorities

- Persistent and portable artifact contracts without serializing ML objects in
  the core.
- Strong replay and bundle validation across prediction caches, artifacts and
  data envelopes.
- Production-shaped host adapters and C ABI contracts.
- Research provenance export roadmap: W3C PROV plus Workflow Run RO-Crate,
  derived from validated DAG-ML lineage and bundles.

## Engineering Rules

- Prefer small, validated slices that move the final product forward.
- Do not weaken OOF, fold, group, repetition, augmentation-origin or refit
  leakage checks for convenience.
- Do not touch `nirs4all` core code.
- Use existing crate patterns and tests.
- Keep JSON compatibility unless a schema version/migration policy is updated.
- Run targeted tests first, then `cargo fmt --check`, `cargo clippy
  --workspace --all-targets -- -D warnings`, `cargo test --workspace`, and
  `python3 scripts/validate_contracts.py`.
