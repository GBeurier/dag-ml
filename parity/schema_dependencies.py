"""Offline, repository-confined JSON Schema dependency closure."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urldefrag, urlsplit


class SchemaDependencyError(ValueError):
    """Raised when a local schema dependency cannot be resolved safely."""


@dataclass(frozen=True)
class SchemaDependencyClosure:
    """Canonical repository paths plus external-reference cycles encountered."""

    paths: tuple[str, ...]
    cycles: tuple[tuple[str, ...], ...]


def _load_schema(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, member in pairs:
            if key in value:
                raise SchemaDependencyError(
                    f"schema {path} contains duplicate object key {key!r}"
                )
            value[key] = member
        return value

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (OSError, json.JSONDecodeError) as error:
        raise SchemaDependencyError(f"cannot load schema {path}: {error}") from error
    if not isinstance(value, dict):
        raise SchemaDependencyError(f"schema {path} must be a JSON object")
    return value


def _ensure_confined_schema_path(
    repository_root: Path, schema_root: Path, path: Path
) -> Path:
    repository_root = repository_root.resolve()
    schema_root_lexical = Path(os.path.abspath(schema_root))
    path_lexical = Path(os.path.abspath(path))
    try:
        path_lexical.relative_to(repository_root)
        path_lexical.relative_to(schema_root_lexical)
    except ValueError as error:
        raise SchemaDependencyError(
            f"schema reference escapes {schema_root_lexical}: {path}"
        ) from error
    cursor = repository_root
    for part in path_lexical.relative_to(repository_root).parts:
        cursor /= part
        if cursor.is_symlink():
            raise SchemaDependencyError(
                f"schema reference traverses symbolic link: {cursor}"
            )
    resolved = path_lexical.resolve()
    schema_root_resolved = schema_root_lexical.resolve()
    try:
        resolved.relative_to(repository_root)
        resolved.relative_to(schema_root_resolved)
    except ValueError as error:  # pragma: no cover - lexical symlink guard is primary
        raise SchemaDependencyError(
            f"resolved schema reference escapes {schema_root_resolved}: {resolved}"
        ) from error
    if not resolved.is_file():
        raise SchemaDependencyError(f"schema dependency is missing: {resolved}")
    return resolved


def _collect_external_refs(value: Any, *, source: str) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        if "$ref" in value:
            ref = value["$ref"]
            if not isinstance(ref, str):
                raise SchemaDependencyError(f"schema {source} has a non-string $ref")
            external, _fragment = urldefrag(ref)
            if external:
                refs.append(external)
        for member in value.values():
            refs.extend(_collect_external_refs(member, source=source))
    elif isinstance(value, list):
        for member in value:
            refs.extend(_collect_external_refs(member, source=source))
    return refs


def schema_dependency_closure(
    repository_root: Path,
    seed_paths: Iterable[str],
    *,
    schema_directory: str = "docs/contracts",
) -> SchemaDependencyClosure:
    """Resolve every external ``$ref`` by local ``$id`` or confined path.

    Fragment-only references do not add files. External cycles are recorded and
    traversed once; unresolved ids, missing files, symlinks and path escapes fail.
    """

    root = repository_root.resolve()
    schema_root = Path(os.path.abspath(root / schema_directory))
    try:
        schema_root.relative_to(root)
    except ValueError as error:
        raise SchemaDependencyError(
            f"schema directory escapes repository root: {schema_root}"
        ) from error
    cursor = root
    for part in schema_root.relative_to(root).parts:
        cursor /= part
        if cursor.is_symlink():
            raise SchemaDependencyError(
                f"schema directory traverses symbolic link: {cursor}"
            )
    schema_root_resolved = schema_root.resolve()
    try:
        schema_root_resolved.relative_to(root)
    except ValueError as error:  # pragma: no cover - lexical symlink guard is primary
        raise SchemaDependencyError(
            f"resolved schema directory escapes repository root: {schema_root_resolved}"
        ) from error
    if not schema_root.is_dir():
        raise SchemaDependencyError(f"schema directory is missing: {schema_root}")

    id_to_path: dict[str, str] = {}
    schemas: dict[str, dict[str, Any]] = {}
    for path in sorted(schema_root.glob("*.schema.json")):
        resolved = _ensure_confined_schema_path(root, schema_root, path)
        relative = resolved.relative_to(root).as_posix()
        schema = _load_schema(resolved)
        schema_id = schema.get("$id")
        if not isinstance(schema_id, str) or not schema_id:
            raise SchemaDependencyError(f"schema {relative} has no non-empty $id")
        schema_id, _fragment = urldefrag(schema_id)
        previous = id_to_path.get(schema_id)
        if previous is not None and previous != relative:
            raise SchemaDependencyError(
                f"schema $id {schema_id!r} is duplicated by {previous} and {relative}"
            )
        id_to_path[schema_id] = relative
        schemas[relative] = schema

    def canonical_seed(raw: str) -> str:
        if not isinstance(raw, str) or not raw:
            raise SchemaDependencyError("schema seed path must be non-empty text")
        if "\\" in raw or Path(raw).is_absolute():
            raise SchemaDependencyError(f"schema seed path is unsafe: {raw!r}")
        resolved = _ensure_confined_schema_path(root, schema_root, root / raw)
        relative = resolved.relative_to(root).as_posix()
        if relative not in schemas:
            raise SchemaDependencyError(
                f"schema seed is outside the indexed schema set: {relative}"
            )
        return relative

    def resolve_ref(source: str, target: str) -> str:
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            dependency = id_to_path.get(target)
            if dependency is None:
                raise SchemaDependencyError(
                    f"schema {source} has unresolved external $ref {target!r}"
                )
            return dependency
        if parsed.query:
            raise SchemaDependencyError(
                f"schema {source} has unsupported query in relative $ref {target!r}"
            )
        raw_path = unquote(parsed.path)
        if not raw_path or "\\" in raw_path or Path(raw_path).is_absolute():
            raise SchemaDependencyError(
                f"schema {source} has unsafe relative $ref {target!r}"
            )
        resolved = _ensure_confined_schema_path(
            root, schema_root, (root / source).parent / raw_path
        )
        relative = resolved.relative_to(root).as_posix()
        if relative not in schemas:
            raise SchemaDependencyError(
                f"schema {source} references non-indexed schema {relative}"
            )
        return relative

    states: dict[str, int] = {}
    stack: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def visit(relative: str) -> None:
        state = states.get(relative, 0)
        if state == 2:
            return
        if state == 1:
            start = stack.index(relative)
            cycles.add(tuple([*stack[start:], relative]))
            return
        states[relative] = 1
        stack.append(relative)
        for target in _collect_external_refs(schemas[relative], source=relative):
            visit(resolve_ref(relative, target))
        popped = stack.pop()
        if popped != relative:  # pragma: no cover - defensive stack invariant
            raise SchemaDependencyError("schema dependency traversal stack corrupted")
        states[relative] = 2

    for seed in sorted({canonical_seed(path) for path in seed_paths}):
        visit(seed)
    return SchemaDependencyClosure(
        paths=tuple(sorted(states)),
        cycles=tuple(sorted(cycles)),
    )


def missing_schema_dependencies(
    repository_root: Path, artifact_paths: Iterable[str]
) -> set[str]:
    """Return transitive schema paths absent from an artifact-path collection."""

    paths = set(artifact_paths)
    seeds = sorted(path for path in paths if path.endswith(".schema.json"))
    closure = schema_dependency_closure(repository_root, seeds)
    return set(closure.paths) - paths


def with_transitive_schema_dependencies(
    repository_root: Path, artifacts: Mapping[str, str]
) -> dict[str, str]:
    """Copy an artifact manifest and add every transitive schema dependency."""

    expanded = dict(artifacts)
    seeds = sorted(path for path in expanded if path.endswith(".schema.json"))
    closure = schema_dependency_closure(repository_root, seeds)
    for path in closure.paths:
        expanded.setdefault(path, "schema_dependency")
    return expanded
