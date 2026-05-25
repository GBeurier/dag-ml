# Generated Demonstrators

This directory contains deterministic outputs from example generators.

## sklearn complex OOF

Regenerate with:

```bash
python examples/sklearn_complex_oof_demo.py
cargo run -p dag-ml-cli -- validate-oof-campaign examples/generated/sklearn_complex_oof_campaign.json
```

The fixture is independent from `nirs4all`. It demonstrates repeated
observations, group-safe OOF, train-only augmentation, branch model variants,
heterogeneous merge variants using predictions plus original data, OOF-based
selection and final refit reporting.
