# ADR-16: Artifact serialization security

**Status**: accepted (2026-05-29)
**Blocks**: workstream A (bundle), workstream E (bridge)

## Context

Fitted models are persisted as artifacts inside `.n4a` bundles. The common Python serializers — `pickle` and `joblib` — execute arbitrary code on load. Loading an untrusted bundle is therefore a remote-code-execution vector. Replay and predict load artifacts by definition, so this is on the hot path.

## Decision

1. **Declared backend per artifact** — bundle metadata records the serialization backend for every artifact: `joblib`, `pickle`, `sklearn_estimator_dict`, `onnx`, `rds`, `torch_state_dict`, etc.
2. **Default-deny on code-bearing backends** — `dag-ml predict` / replay **refuses** to load a `pickle` / `joblib` / code-bearing `rds` artifact unless the caller passes an explicit opt-in: `--allow-pickle` (CLI) or `allow_unsafe_artifacts=True` (Python). The default raises `security/unsafe_artifact_refused` (ADR-11) with a remediation hint.
3. **Recommended code-free backends** — `sklearn_estimator_dict` (params + fitted arrays, no pickled callables), `onnx`, and framework-native state dicts are safe to load without the flag. The sklearn process adapter prefers `sklearn_estimator_dict` where the estimator supports round-tripping its fitted state as plain arrays.
4. **Dual validation** — the loader checks both the per-artifact content fingerprint (already in `bundle.rs`) and the declared backend before loading. A backend that doesn't match the file's detected format is refused.
5. **Bundle signing (deferred)** — Sigstore/cosign signing is the intended hardening path but is **deferred** to a follow-up release (descoped). This ADR reserves a `signature` field in the bundle schema now so adding signing later is non-breaking.

## Consequences

- The bundle schema gains `artifacts[].serialization_backend` (required) and a reserved optional `signature` field.
- Workstream A's bundle loader implements the default-deny gate; workstream E's artifact bridge tags each persisted artifact with its backend.
- `SECURITY.md` documents the artifact-loading trust model next to ADR-13 (the process-adapter half).

## Risk

- Default-deny breaks naïve "just `joblib.dump` everything" workflows. This is intentional and mirrors `numpy.load(allow_pickle=False)` and `torch.load(weights_only=True)` precedent. The error names the `--allow-pickle` flag and recommends migrating to `sklearn_estimator_dict`.
