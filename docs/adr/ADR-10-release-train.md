# ADR-10: Cross-repo release train

**Status**: accepted (2026-05-28)
**Blocks**: workstream C (release process), workstream D (Python packaging).

## Context

dag-ml depends on dag-ml-data; nirs4all depends on dag-ml + dag-ml-data (eventually). Without a scripted release order, version pinning drifts (Codex flagged this as the "cross-repo release coordination" risk).

## Decision

The release train is a fixed five-step sequence, scripted and CI-gated:

1. **`dag-ml-data` releases first**. `cargo-release` cuts a tag, publishes the four `dag-ml-data-*` crates to crates.io, builds and uploads the Python wheel (`dag-ml-data-py`) to PyPI, and runs the cross-header parity test against `dag-ml` at HEAD.

2. **`dag-ml`'s pinned `dag-ml-data` version bumps**. A scripted PR (opened by the release workflow) updates the workspace `Cargo.toml`'s `dag-ml-data = { version = "X.Y.Z" }` constraint, runs the green gate, and merges automatically if the gate is green.

3. **`dag-ml` releases**. Same scripted path: `cargo-release` → crates.io for the four `dag-ml-*` crates → PyPI wheel for `dag-ml-py` → CHANGELOG entry referencing ADR-10.

4. **`nirs4all`'s pinned dag-ml + dag-ml-data versions bump**. Same scripted-PR pattern; nirs4all's CI gates on dag-ml ≥ X.Y.Z AND dag-ml-data ≥ X.Y.Z.

5. **`nirs4all` releases** (independent cadence; usually one release per train run).

### CI gates

- Each release tag triggers `release-crates.yml` in the repo being released. The workflow validates release metadata, validates the Cargo publish plan, checks that the `v*` tag matches the workspace version and publishes crates in dependency order.
- `release-crates.yml` accepts prerelease SemVer tags such as `v0.1.0-alpha.1`; this is required while `dag-ml` and `dag-ml-data` are still in alpha.
- `cargo publish --dry-run` runs in CI through `scripts/release/check_publish_plan.py --dry-run`.
- A "release ready" PR check verifies CHANGELOG, ADR delta, and ABI snapshot are present.

### Hotfix path

For a single-repo hotfix (e.g. dag-ml only, leaving dag-ml-data unchanged), the release script accepts `--skip-data` and runs steps 3–5 directly. The CHANGELOG entry must justify the skip.

## Consequences

- `scripts/release/` lands in both repos with per-repo `release-crates.yml` GitHub Actions workflows for crates.io publication.
- CONTRIBUTING.md documents the train and the hotfix exception.
- Version pinning becomes a hard CI constraint; manual edits are reverted by the bot.

## Risk

- A breaking change in `dag-ml-data` that `dag-ml` cannot adopt blocks the whole train. The release script surfaces this with a clear "dag-ml does not compile against dag-ml-data X.Y.Z" error and refuses to proceed. Resolution: either revert the breaking change or land the dag-ml-side migration first.
