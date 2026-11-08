# Genuine schema2 attached calibration fixture

Immutable selected bytes imported only after the root published
`/tmp/phase2a-b-calibration-fixture-ready.json`. The producer used current published
DAG 0.3.44 owner APIs and actual Methods ABI 2.20, library SHA256
`05ae1a2e82491439f0e71ada0ea500ed79d03af227a11c4a2dbb317a99558529`.

This is NEW controlled synthetic training (40 development samples), followed by
actual native PREDICT on 48 disjoint heldout calibration samples, then
`calibrate_attached_training_replay_with_derived_context` and
`TrainingOutcome::to_portable_predictor_package`. The sole target is `protein`.
The raw fitted model SHA256 is unchanged before/after calibration:
`6a721ee4c688a1b4b38fb4bc43e94b621d7134249b475772c020adee96b204a9`.
No old archive builder was used. Owner JSON bytes are preserved exactly.

`source/` retains the actual producer sources and synthetic input generator;
`provenance/` and `outputs/producer_facts.json` describe generation and original
qualification. `import-inventory.json` records every imported byte hash. No
compiled generator or dependency archive is shipped here. Producer source is
provenance, not a runtime dependency of package tests.

These fixtures support default owner contract tests without a native library.
Original native generation is evidence of that producer run only, not of the new
B port, a new IO/DMD same-snapshot adapter, installed host facades, platform parity
or generalization. Native B-port and actual IO/DMD qualification are separate.
The source contains one target; mutations test target-name/shape refusal, not a
claim that this fixture exercises a multi-target permutation.

Calibration stays inside the actual outcome/package with a bundle reference.
There is no calibration facet, standalone calibration member or new profile.
