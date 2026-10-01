# Archive V2 native-portable replay contract

Status: ADR-23 native-portable contract with the additive ADR-28 Methods
RolePipeline profile. DAG-ML assembles and validates the replay closure; Core
owns the archive writer and bounded reader.

Archive V2 is the strict native-only successor to the frozen Archive V1 wire.
Its dispatch identity is exact:

- `schema_version`: `2`
- `profile`: `nirs4all.archive_workspace.v2`
- writer owner: `nirs4all-core`
- writer ID: `nirs4all-core.archive_workspace_writer.v2`

The V2-capable reader is dual-read: exact V1 remains immutable and exact V2 is
accepted; future versions and version-family mixing are refused before
extraction. A V1-to-V2 migration is copy-on-write, retains the raw V1 source and
creates a new archive identity. The frozen files under `archive-v1/` are not a
template to rewrite, and their validator remains byte-identical. W1 conformance
packs are separate governed provenance: independently approved changes to inputs
such as the ADR registry, CI, or `validate_contracts.py` require canonical pack
regeneration by the owning release workflow. They must not be reconciled by
manually patching individual hashes, and that regeneration does not modify the
frozen Archive V1 wire.

## Replay closure

The six mandatory DAG-ML members are distinct and inventory-closed:

1. `PortablePredictorPackage` V2 at the fixed member path
   `dagml/portable_predictor_package.json`;
2. exact `GraphSpec` V1;
3. `ExecutionBundle` V2;
4. `TrainingOutcome` V2;
5. prediction-cache payload set V2; and
6. `ScoreSet` V2.

The package and every runtime V2 reference declare
`producer_port_required: true`. Package V2 embeds Bundle V2; Package V1 remains
closed, rejects V2-only keys even when their value is null, and embeds Bundle
V1. Archive V1 therefore pairs only with Package V1, while Archive V2 pairs only
with Package V2.

Portable package V2 owns `conformal_calibration` and
`conformal_calibration_replay`. Archive V2 requires the older archive-level
`payloads.conformal` slot to be null. The null slot is an explicit ownership
marker, not an equivalent second representation.

## Native-only boundary

Archive V2 P0 is intentionally a **Methods-only** native profile. It accepts
only packages whose `fitted_artifact_mode` is `portable_required`, whose
artifact bindings all use `native_portable`, and whose every refit artifact is
a raw, plugin-free `n4m_model`. `payloads.host_artifacts` is exactly empty. A
host sidecar, external reference, Python fallback, or non-Methods refit artifact
is a pre-write refusal, not a degraded replay mode. This does not declare that
all future native-portable archive profiles must be Methods-only; widening the
P0 policy requires a later policy ADR/profile. ADR-28 adds the bounded
RolePipeline family described below; it does not loosen the N4MM declarations.

At least one N4MM member is required. Each N4MM reference is format version 1
for historical/raw PLS or version 2 for the native SNV -> SG smooth -> PLS pipeline,
Methods ABI major 2 and an additive `abi_min_minor`, carries its package
`artifact_id`, and points to exact raw
bytes at a safe `methods/*.n4mm` path. The archive N4MM IDs exactly cover the
package's raw refit artifacts and each ZIP member byte-equals the corresponding
`ExecutionBundle.raw_artifact_payloads` entry. The native binary semantic
profile is `n4mm_raw_sha256`: its semantic fingerprint is exactly the raw
N4MM SHA-256, rather than a fabricated JSON-canonicalization digest. Raw
SHA-256, size and semantic identity match the closed inventory; this matching
semantic identifier never substitutes for the raw model payload.
Writers derive the minimum from the payload capability: historical PLS N4MM is
2.0+, imported-linear/Ridge N4MM is 2.3+, and pipeline N4MM v2 is 2.5+. An absent minor remains readable
only for the historical PLS profile; it never defaults a Ridge payload to 2.3.

## Additive RolePipeline transport

`payloads.methods.role_pipelines` is optional and absent on N4MM-only exports.
The nonempty union of N4MM and RolePipeline declarations exactly covers every
package artifact binding, refit record and raw payload. Pure RolePipeline exports
use `n4mm: []`; no placeholder N4MM is created. Each RolePipeline reference uses
the closed ADR-28 fields, content-addressed `artifacts/<sha256>.json` member and
`dagml_methods_role_pipeline_raw_sha256` semantic profile.

The wrapper schema is `dagml.methods.regression.v1`, the accepted signed plugin
is `dagml.methods.wasm.regression` version `1.0.0`, and the original package
retains its controller trust and exact recipe/state identities. DAG-ML validates
the bounded wrapper; Methods hydrates opaque N4ME states. Core checks storage
bounds/inventory and exposes bytes. Consumers then call
`validate_archive_v2_portable_payloads(manifest, package, members)` before any
callbacks. It accepts only the exact standalone canonical transport, including
all six companions. Python `replay_loaded_predictor_package` additionally accepts
`trusted_controller_manifests`; archive consumers supply them explicitly.

The process-separated Node/web qualification is:

```bash
NIRS4ALL_CORE_WASM_MODULE=/path/to/core-wasm/src/index.js \
  node scripts/smoke_wasm_multimodal_methods_hpo.mjs \
  /path/to/dagml/pkg-node /path/to/methods/dist /tmp/five-model-proof.json
```

Repeat with `/path/to/dagml/pkg-web`. A comma-separated Core module list also
qualifies extracted public package consumers. The proof records whether Core
transport actually ran; without the environment variable the campaign checks
DAG closure and package replay only. Its four numeric projections plus OOF meta
model are five genuine fitted models, not a canonical N-D qualification.

## Contract artifacts and gate

- `archive_workspace_manifest.v2.schema.json` is the closed Draft 2020-12
  manifest schema.
- `fixtures/positive/native_portable_replay.json` is the canonical manifest.
- `fixtures/positive/portable_predictor_package.native_methods.v2.json` is the
  canonical Package V2 member with raw, plugin-free Methods artifacts and must
  pass DAG-ML's semantic `PortablePredictorPackage::from_json` owner validator.
- `fixtures/negative/refusals.v2.json` freezes version-mixing, host-state,
  hash/tamper, future/unknown-field and producer-port refusals.
- `scripts/validate_archive_v2_contract.py` validates schema shape, semantic
  closure, the Package V1/V2 boundary and materialized bounded ZIPs.

Run:

```bash
python3 scripts/validate_archive_v2_contract.py
python3 -m unittest tests.test_archive_v2_contract
```

The validator is also called additively by `scripts/validate_contracts.py`.
