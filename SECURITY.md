# Security Policy

## Reporting a vulnerability

Please report security issues **privately** via GitHub's "Report a vulnerability"
(Security → Advisories) on the `dag-ml` repository, or by emailing the maintainer
listed in `Cargo.toml`. Do not open a public issue for a vulnerability.

We aim to acknowledge a report within 5 business days and to agree on a
disclosure timeline before any public discussion.

## Supported versions

`dag-ml` is pre-1.0 (`0.2.x`). Security fixes target the latest
published version only until a stable line exists.

| Version | Supported |
| ------- | --------- |
| 0.2.x | ✅ latest only |

## Trust model — two host-code surfaces

`dag-ml` orchestrates host code. Two surfaces carry the highest risk and have
dedicated decision records:

- **Process / in-process controllers run host code.** Loading and running a
  `ControllerManifest` is equivalent to running its code. The runtime gates
  this behind an executable allowlist, environment sanitization, per-task
  timeout/kill, and tempdir isolation, with an optional sandbox hook. See
  [ADR-13](docs/adr/ADR-13-process-adapter-security.md).
- **Artifacts can execute code on load.** `pickle`/`joblib` model artifacts are
  arbitrary-code-execution payloads. Replay and predict **default-deny**
  code-bearing artifacts; loading one requires an explicit `--allow-pickle`
  opt-in. Prefer code-free backends (`sklearn_estimator_dict`, ONNX, native
  state dicts). See [ADR-16](docs/adr/ADR-16-artifact-security.md).

When triaging a report, please indicate which surface (controller execution,
artifact loading, contract validation, or other) it concerns.

## Hardening guidance for operators

- Set `DAGML_ADAPTER_ALLOWLIST` to the minimal set of trusted executables.
- Keep `--allow-pickle` off unless you fully trust the bundle's origin.
- Enable `DAGML_ADAPTER_SANDBOX` in multi-tenant or untrusted-input deployments.
