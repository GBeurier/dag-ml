#!/usr/bin/env python3
"""Validate artifact-facets captures against the v1 JSON Schema (shape only).

By default, both committed host and native_role captures are required and
validated with the Draft 2020-12 validator. ``--capture`` selects one instance.
The gate checks structure only: ownership, link directions, fingerprints,
requirement meaning and anything runtime-related remain with the Rust owner
validators.  No remote reference is ever retrieved.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "docs/contracts/artifact_facets.v1.schema.json"
DEFAULT_CAPTURE = ROOT / "crates/dag-ml-core/tests/fixtures/artifact_facets/capture/facets.json"
NATIVE_ROLE_CAPTURE = ROOT / "crates/dag-ml-core/tests/fixtures/artifact_facets/native_role/capture/facets.json"
# Both committed instances are required; a missing one fails the default gate.
DEFAULT_CAPTURES = (DEFAULT_CAPTURE, NATIVE_ROLE_CAPTURE)


class ArtifactFacetsContractError(Exception):
    """Raised when the gate cannot run or the instance is refused."""


try:
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError
    from referencing import Registry
    from referencing.exceptions import NoSuchResource
except ImportError as error:  # explicit failure, never a skip
    raise SystemExit(
        f"artifact facets gate: missing dependency ({error}); "
        "install with: python -m pip install -r scripts/requirements-contracts.txt"
    ) from error


def deny_retrieval(uri: str) -> Any:
    """Registry retrieval hook: remote or unknown resources are never fetched."""
    raise NoSuchResource(ref=uri)


def load_json(path: Path, label: str) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ArtifactFacetsContractError(f"{label} is absent: {path}") from error
    except (OSError, UnicodeDecodeError) as error:
        raise ArtifactFacetsContractError(f"{label} is unreadable: {path}: {error}") from error
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ArtifactFacetsContractError(f"{label} is malformed JSON: {path}: {error}") from error


def build_validator(schema: Any) -> Draft202012Validator:
    """Check the schema itself and return a validator with network retrieval denied."""
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise ArtifactFacetsContractError(
            f"schema is not well-formed at schema path {format_path(error.absolute_path)}: {error.message}"
        ) from error
    return Draft202012Validator(schema, registry=Registry(retrieve=deny_retrieval))


def format_path(path: Any) -> str:
    items = list(path)
    return "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in items)


def validate_instance(validator: Draft202012Validator, instance: Any) -> list[str]:
    """Return one message per shape violation; empty list means accepted."""
    try:
        errors = sorted(validator.iter_errors(instance), key=lambda e: [str(p) for p in e.absolute_path])
    except Exception as error:  # unresolved or denied references must fail loudly
        raise ArtifactFacetsContractError(f"schema reference could not be resolved offline: {error}") from error
    return [
        f"instance {format_path(e.absolute_path)} violates schema {format_path(e.absolute_schema_path)}: {e.message}"
        for e in errors
    ]


def validate_artifact_facets_contract(
    capture_paths: Path | Sequence[Path] = DEFAULT_CAPTURES, schema_path: Path = SCHEMA_PATH
) -> list[Path]:
    """Validate every given capture (all must exist and pass) and return the validated paths."""
    paths = [capture_paths] if isinstance(capture_paths, Path) else list(capture_paths)
    if not paths:
        raise ArtifactFacetsContractError("no capture selected for validation")
    validator = build_validator(load_json(schema_path, "schema"))
    failures: list[str] = []
    for path in paths:
        problems = validate_instance(validator, load_json(path, "capture"))
        if problems:
            shown = "\n  ".join(problems)
            failures.append(
                f"capture {path} refused by schema {schema_path} ({len(problems)} violation(s)):\n  {shown}"
            )
    if failures:
        raise ArtifactFacetsContractError("\n".join(failures))
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="validate only this capture JSON (default: the committed host and native_role captures)",
    )
    args = parser.parse_args(argv)
    selected = DEFAULT_CAPTURES if args.capture is None else (args.capture,)
    try:
        validated = validate_artifact_facets_contract(selected)
    except ArtifactFacetsContractError as error:
        print(f"artifact facets contract FAILED: {error}", file=sys.stderr)
        return 1
    listing = "\n".join(f"  {path}" for path in validated)
    print(f"artifact facets contract OK (shape only): {len(validated)} capture(s) validated\n{listing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
