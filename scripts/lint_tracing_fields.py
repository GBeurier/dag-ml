#!/usr/bin/env python3
"""ADR-12 observability privacy lint.

Enforces two invariants so telemetry never leaks raw data:

1. **Centralization** — `tracing` is used only in
   `crates/dag-ml-core/src/observability.rs`. Every other core module must route
   telemetry through that module's vetted helpers, so there is a single place to
   audit emitted fields.
2. **Field privacy** — the non-test code of `observability.rs` (comments and
   string literals stripped) must not contain any identifier matching
   `data|features|targets|samples|metadata`, and the declared
   `OBSERVABILITY_FIELD_ALLOWLIST` entries must be free of the same pattern.

Span/event fields are identifiers and counts only; raw features, targets, sample
values and metadata never cross the telemetry boundary (ADR-12).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = ROOT / "crates" / "dag-ml-core" / "src"
OBSERVABILITY = CORE_SRC / "observability.rs"

# `sample` (singular) subsumes `samples`, so e.g. `sample_count`/`sample_ids`
# are also rejected, matching the "never sample values" privacy rule.
FORBIDDEN = re.compile(r"\b\w*(?:data|features|targets|sample|metadata)\w*\b")
TRACING_TOKEN = re.compile(r"\btracing\b")


def fail(message: str) -> None:
    raise SystemExit(f"tracing-field lint failed: {message}")


def strip_comments_and_strings(source: str) -> str:
    """Remove `//` line comments and `"..."` string literals (best-effort)."""
    without_comments = re.sub(r"//[^\n]*", "", source)
    without_strings = re.sub(r"\"(?:\\.|[^\"\\])*\"", "", without_comments)
    return without_strings


def non_test_code(source: str) -> str:
    marker = source.find("#[cfg(test)]")
    return source if marker < 0 else source[:marker]


def main() -> None:
    require_observability_present()
    enforce_centralization()
    enforce_field_privacy()
    enforce_allowlist_privacy()
    print("validated ADR-12 observability field privacy")


def require_observability_present() -> None:
    if not OBSERVABILITY.is_file():
        fail(f"missing observability module: {OBSERVABILITY}")


def enforce_centralization() -> None:
    for path in sorted(CORE_SRC.rglob("*.rs")):
        if path == OBSERVABILITY:
            continue
        code = strip_comments_and_strings(path.read_text(encoding="utf-8"))
        if TRACING_TOKEN.search(code):
            rel = path.relative_to(ROOT)
            fail(
                f"{rel} uses `tracing` directly; route telemetry through "
                "crate::observability so emitted fields stay vetted"
            )


def enforce_field_privacy() -> None:
    code = strip_comments_and_strings(non_test_code(OBSERVABILITY.read_text(encoding="utf-8")))
    match = FORBIDDEN.search(code)
    if match:
        fail(
            f"observability.rs code references forbidden telemetry token "
            f"`{match.group(0)}`; spans must carry identifiers/counts only"
        )


def enforce_allowlist_privacy() -> None:
    source = OBSERVABILITY.read_text(encoding="utf-8")
    block = re.search(
        r"OBSERVABILITY_FIELD_ALLOWLIST[^=]*=\s*&\[(.*?)\];",
        source,
        re.DOTALL,
    )
    if not block:
        fail("could not find OBSERVABILITY_FIELD_ALLOWLIST declaration")
    entries = re.findall(r"\"([^\"]+)\"", block.group(1))
    if not entries:
        fail("OBSERVABILITY_FIELD_ALLOWLIST is empty")
    for entry in entries:
        if FORBIDDEN.search(entry):
            fail(f"allowlisted field `{entry}` matches a forbidden data pattern")


if __name__ == "__main__":
    main()
