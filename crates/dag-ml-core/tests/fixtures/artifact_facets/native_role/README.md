# Genuine native RolePipeline and target-bound child-refit fixtures

Imported only after `/tmp/phase2a-c-owner-fixtures-ready.json` was published.
`provenance.json` is the immutable producer receipt (SHA-256
`dcb8246a7d60b187970a0bd3b5b868f6807b61b6e660a5b4ddd09592cf0ac7e3`).
The source producer used the published `dag-ml-core` 0.3.44 crate, native feature,
`n4m` 0.4.1 dynamic binding, and the actual Methods 1.3.14 / ABI 2.20 library:
`libn4m.so.2.20.0`, SHA-256
`05ae1a2e82491439f0e71ada0ea500ed79d03af227a11c4a2dbb317a99558529`.
Published owner source commit: `01226120552529aad45e5ba60d3f6327894adc67`.
Generation used Rust 1.96; the current-source Rust 1.88 default tests independently
load and validate these exact bytes before exercising the new facet API.

The producer called actual owner APIs, before any archive assembly:

1. `execute_training` and native `RolePipeline::fit/export_states`.
2. `TrainingOutcome::to_portable_predictor_package` with native portable mode.
3. `PortableRefitRecipe::derive_from_package` and
   `derive_portable_full_refit_target_plan` for distinct target sample IDs/data.
4. Fresh `execute_portable_full_refit`, then `build_portable_refit_package_v3`.

`training_outcome.json`, `predictor_package.json` and `child_refit_v3.json` retain
exact owner exports. The raw wrapper files equal the root packages' raw byte
arrays; their `.n4me` sidecars equal the exact opaque nested state arrays. The
producer imported/re-exported these states through actual Methods owner APIs
and independently reproduced all 32 exports byte-for-byte. This is the real
single-step PLS RolePipeline family; it is not a fabricated N4ME header or an
independent bare-N4ME artifact contract. Generic facet validation does not
numerically decode these bytes. The optional facet N4MM inspector leaves this
unsupported RolePipeline family explicitly unverified.

`generator_main.rs.txt`, `generator_helpers.rs.txt`,
`generator_Cargo.toml.txt` and `generator_*.json` retain the exact generator
source and its input fixtures. They are provenance snapshots, not workspace
manifests, dependencies or compiled test modules. To reproduce, restore those
snapshots into a separate temporary Cargo project with its own lock/target,
using the published dependencies recorded in the manifest snapshot and receipt.
The historical exact commands and source hashes are in `provenance.json`.
No vendor tree, Cargo home, temporary source replacement or compiled output is
included here. No old archive builder/reader was used.

All numeric inputs are synthetic. Feature/target row sidecars and source/target
requests, relations and influence are retained. Their schema/plan identifiers
are generic coordinator fixture values, not IO-issued logical dataset proof.
Neither these fixtures nor successful raw hashing establish IO identity,
IO-to-DMD/DAG mapping, evaluation population parity, or independent child scores.
The native library's embedded git-revision query was empty; this unavailable
metadata is recorded separately from the observed source checkout revision.

Default regressions derive duplicate references to the same genuine raw state
in memory to test same-SHA physical deduplication. They do not claim another fit.
The original fixture bytes remain unchanged.

`capture/` is emitted by the current direct `artifact_facets` example from these
exact outcome/package inputs. It includes content-addressed physical raw-state
members. The DAG-owned model facet owns the root and native children exactly
once; no separate Methods facade duplicates them. `links.json` is diagnostic
only and cannot be deserialized into a live facet verdict.

For regeneration, the flattened snapshots map back to these original names:
`generator_main.rs.txt` -> `src/main.rs`, `generator_helpers.rs.txt` ->
`src/helpers.rs`, `generator_Cargo.toml.txt` -> `Cargo.toml`, and each
`generator_<name>.json` -> `fixtures/package/<name>.json`. The preserved helper
source supplies the exact original include paths. Use a separate temporary
project and the actual pinned Methods library; do not add these fixture
producer dependencies to the product workspace.
