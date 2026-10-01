# ADR-28: Methods RolePipeline transport in Archive V2

**Status**: accepted (2026-10-01)
**Extends**: ADR-23 with a bounded native codec; its N4MM profile stays unchanged.
**Scope**: HPO-01 complete predictor-package persistence and PREDICT replay.

## Decision

Archive V2 may add `payloads.methods.role_pipelines`. The field is omitted for
N4MM-only packages, preserving their existing manifest and member bytes. The
nonempty union of N4MM and RolePipeline declarations exactly covers the signed
package's artifact bindings, refit records and detached raw payload map. Mixed
packages are permitted; a pure RolePipeline package has `n4mm: []`.

Each RolePipeline declaration is closed and contains `artifact_id`,
`kind: methods_role_pipeline`, `owner: dag-ml`, `format_version: 1`,
`member_path: artifacts/<raw-sha256>.json`, `raw_sha256`,
`semantic_fingerprint: <raw-sha256>` and
`semantic_profile: dagml_methods_role_pipeline_raw_sha256`.
The existing signed package retains plugin identity, controller manifests and
trust requirements. The archive declaration does not duplicate or override them.

DAG-ML accepts only the RAW wrapper `dagml.methods.regression.v1`, produced by
plugins `dagml.methods.wasm.regression`, `dagml.methods.r.regression`,
`dagml.methods.octave.regression` and `dagml.methods.native.regression`, each
version `1.0.0`. Their controller identities and explicit trust manifests remain
distinct; sharing a native state codec does not authorize another host. It validates the exact
node and parameter identity, unique ordered feature/target names, bounded native
Methods recipe and one opaque N4ME state per step. It never interprets estimator
bytes. Native Methods hydration owns state compatibility and numerical behavior.
DAG-ML recomputes the full ordered recipe from the effective graph operator and
overlays selected node parameters only on its final step. A copied parameter
fingerprint cannot authorize different step methods or parameters. The runtime
controller also compares saved ordered feature names with independently resolved
current data/OOF names before numerical prediction; a width match is insufficient.
Unknown wrapper fields, other codecs/plugins, host artifacts, Python pickle,
external paths, missing/extra IDs and byte/hash/URI mismatches are refused.

Core owns bounded ZIP storage, manifest references and inventory integrity. Its
opaque reader exposes the manifest and members; it does not lower recipes or
replay controllers. Before callbacks, consumers invoke DAG-ML's native
`validate_archive_v2_portable_payloads`, which reconstructs the canonical
standalone transport and checks every original document, reference and raw byte.
This profile excludes optional workspace extras. Runtime replay separately
compares signed controller manifests with explicit trusted installed manifests.
Neither archive integrity nor a raw SHA identifies a trusted implementation.

## Replay and qualification

The package remains PREDICT/EXPLAIN-only under ADR-21/25. Replay supplies named
current-source inputs, resolves the original DAG topology and dependencies,
hydrates every required fitted artifact, and releases all states on success
or failure. It cannot fit held-out inputs or flatten a multi-model graph into
a single estimator. Export preserves the original package/outcome fingerprints,
OOF cache evidence, selected parameters and all five model identities.

`scripts/smoke_wasm_multimodal_methods_hpo.mjs` captures real four-source numeric
projections and an OOF meta Ridge, assembles the five-model closure, and optionally
writes it with Core's public WASM API. A separate consumer process reads the
`.n4a`, validates the closure and predicts with fitting disabled. Node and web
DAG-ML packages run the same campaign. The SDK exposes the matching explicit
Python archive read/write/replay facade. These numerical projections do not
qualify canonical N-D encoding, every estimator, R or Octave transport.

N4MM is not relabeled, and no dummy N4MM is inserted. Generic RolePipeline RAW
states are not silently converted to the narrow N4MM descriptor profile.
The authored Octave adapter uses the same wrapper through the public Methods
RolePipeline MEX. Its plugin is bound to `controller:methods.octave.regression`;
its native runtime qualification remains pending. See
[the adapter guide](../OCTAVE_METHODS_ROLE_ADAPTER.md).
Published package versions and tags remain immutable.
