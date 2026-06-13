# ADR-09: Docs stack

**Status**: accepted (2026-05-28)
**Blocks**: workstream C (maturity).

## Context

dag-ml and dag-ml-data both ship raw markdown today (`docs/*.md`). The integration plan demands a hosted, contributor-friendly site that links to API reference. Choice: Sphinx (Python-ecosystem-friendly), mdBook (Rust-ecosystem-friendly), Docusaurus (JS-ecosystem-friendly).

## Decision

**Sphinx + MyST for prose; rustdoc/docs.rs for the API reference; one landing page per repo linking both.**

Rationale:
- Sphinx + MyST renders the existing `.md` files unchanged (no rewrite required).
- nirs4all already ships Sphinx; reusing the same toolchain reduces the maintenance footprint and lets bridge contributors learn once.
- rustdoc auto-publishes on crates.io release (workstream D); pointing the landing page at `https://docs.rs/dag-ml-core/latest/` gives a free always-current API reference.
- mdBook would split the contributor experience (Rust API vs prose vs Python wheels in three places). Sphinx unifies it.

### Concrete layout

```
docs/
├── conf.py                  # Sphinx config (MyST + sphinx_design + sphinx_copybutton)
├── index.md                 # landing page; links to ADRs / STATUS / API ref
├── installation.md
├── architecture.md          # symlink or include of ARCHITECTURE.md
├── adr/                     # numbered ADRs (this directory)
├── contracts/               # JSON schemas (existing)
├── design/source/           # historical specs (existing)
└── _build/                  # gitignored
```

### Hosting

- GitHub Pages on tag-push from `main` (workstream C task 5 + ADR-10 release train).
- rustdoc auto-published via crates.io on `cargo publish` (ADR-10).

## Consequences

- Workstream C task 5 adds Sphinx config + GitHub Pages workflow per repo.
- nirs4all's existing Sphinx pipeline serves as the template (the sibling `nirs4all` repo's `docs/source/conf.py`).
- The site replaces `docs/TOC.md` as the canonical navigation entry point (TOC.md remains in-repo as the source).

## Risk

- Sphinx requires Python build dependencies even though the projects are Rust. CI installs them in the docs job only — the build/test gate stays Rust-only. Acceptable.
