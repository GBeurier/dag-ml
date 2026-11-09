# Package-local test fixtures

These small fixtures exercise `dag-ml-core` behavior after `cargo package`
without relying on sibling workspace files.  They are deliberately separate
from the repository-wide contract corpus.  Tests that assert the published
schemas or generated workspace examples are gated behind
`dag_ml_workspace_contract_fixtures`; run them in the workspace with:

```bash
RUSTFLAGS='--check-cfg=cfg(dag_ml_workspace_contract_fixtures) --cfg dag_ml_workspace_contract_fixtures' \
  cargo test -p dag-ml-core
```

The package-portability script validates the cfg name but leaves it disabled,
which confirms that an extracted crate needs no repository documents or
generated examples.

`archive/training_outcome_port_explicit.json` is the package-local copy used
by Archive V2 tests that must also compile in an extracted crate.  It is kept
outside the repository-wide W1 corpus and deliberately has no versioned
filename, so the package gate can continue to reject unapproved protocol
fixtures.

Additional package-local byte-for-byte mirrors (the workspace paths remain canonical):

- `realization_evidence.schema.json` ← `docs/contracts/realization_evidence.v1.schema.json`
- `score_set.json` ← `examples/fixtures/score_set.json`
- `training/portable_predictor_package.json` ← `examples/fixtures/training/portable_predictor_package.v1.json`
- `training/training_outcome_refit.json` ← `examples/fixtures/training/training_outcome_refit.v1.json`
- `training/replay/training_replay_outcome_predict.json` ← `examples/fixtures/training/replay/training_replay_outcome_predict.v1.json`
- `training/replay/training_replay_request_predict.json` ← `examples/fixtures/training/replay/training_replay_request_predict.v1.json`
- `training/replay/training_replay_input_envelopes.json` ← `examples/fixtures/training/replay/training_replay_input_envelopes.v1.json`
- `direct_authority/minimal_graph.json` ← `examples/minimal_graph.json`
- `direct_authority/bundle_build_spec_minimal.json` ← `examples/fixtures/bundle/bundle_build_spec_minimal.json`

The direct authority test also reuses the existing exact package-local data envelope, campaign and controller manifests. Recheck copies against their canonical workspace bytes when updating those inputs.
