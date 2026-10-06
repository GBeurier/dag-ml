#!/usr/bin/env python3
"""Validate the ADR-10 Cargo publish order and optional dry-run roots."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib


@dataclass(frozen=True)
class Crate:
    name: str
    manifest: Path
    internal_deps: tuple[str, ...]


def fail(message: str) -> None:
    raise SystemExit(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def dependency_names(table: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for section in ("dependencies", "build-dependencies", "dev-dependencies"):
        values = table.get(section, {})
        if isinstance(values, dict):
            names.update(values)
    for target in table.get("target", {}).values():
        if not isinstance(target, dict):
            continue
        for section in ("dependencies", "build-dependencies", "dev-dependencies"):
            values = target.get(section, {})
            if isinstance(values, dict):
                names.update(values)
    return names


def workspace_crates(repo: Path) -> tuple[str, list[Crate]]:
    root = load_toml(repo / "Cargo.toml")
    workspace = root["workspace"]
    version = workspace["package"]["version"]
    members = workspace["members"]
    workspace_deps = workspace.get("dependencies", {})
    package_names: dict[str, Path] = {}
    manifests: dict[str, dict[str, Any]] = {}

    for member in members:
        manifest_path = repo / member / "Cargo.toml"
        manifest = load_toml(manifest_path)
        package = manifest["package"]
        name = package["name"]
        if package.get("publish") is False:
            continue
        require(
            isinstance(package.get("version"), dict) and package["version"].get("workspace") is True,
            f"{manifest_path}: package.version must inherit workspace version",
        )
        package_names[name] = manifest_path
        manifests[name] = manifest

    crates: list[Crate] = []
    for name, manifest in manifests.items():
        internal = sorted(dependency_names(manifest).intersection(package_names))
        for dep_name in internal:
            dep = workspace_deps.get(dep_name)
            require(isinstance(dep, dict), f"workspace dependency {dep_name} must be a table")
            require(dep.get("path"), f"workspace dependency {dep_name} must declare path")
            require(
                dep.get("version") == version,
                f"workspace dependency {dep_name} must pin version {version}",
            )
        crates.append(Crate(name=name, manifest=package_names[name], internal_deps=tuple(internal)))

    return version, sorted(crates, key=lambda crate: (len(crate.internal_deps), crate.name))


def dry_run_roots(repo: Path, crates: list[Crate]) -> None:
    roots = [crate for crate in crates if not crate.internal_deps]
    require(roots, "publish plan has no independently dry-runnable root crates")
    for crate in roots:
        subprocess.run(
            ["cargo", "publish", "--dry-run", "--allow-dirty", "-p", crate.name],
            cwd=repo,
            check=True,
        )


def package_workspace(repo: Path, version: str) -> None:
    """Package every member without replacing an immutable registry checksum.

    Core's test-only Archive dependency can bring the published dag-ml-core at
    the workspace version into Cargo.lock. Cargo's workspace packaging overlay
    then substitutes a newly packed local archive for that immutable registry
    entry. Package the local core separately in that case; its extracted-package
    test remains mandatory in CI, and dependents resolve the published checksum.
    Before that version is published, retain the normal workspace overlay.
    """
    lock = load_toml(repo / "Cargo.lock")
    collision = any(
        package["name"] == "dag-ml-core"
        and package["version"] == version
        and package.get("source", "").startswith("registry+")
        for package in lock["package"]
    )
    command = ["cargo", "package", "--workspace", "--no-verify"]
    if collision:
        with tempfile.TemporaryDirectory(prefix="dag-ml-core-package-") as target:
            subprocess.run(
                ["cargo", "package", "-p", "dag-ml-core", "--no-verify"],
                cwd=repo,
                env={**os.environ, "CARGO_TARGET_DIR": target},
                check=True,
            )
        command.extend(["--exclude", "dag-ml-core"])
    subprocess.run(command, cwd=repo, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run cargo publish --dry-run for crates without internal dependencies",
    )
    parser.add_argument("--package", action="store_true", help="package every workspace crate")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    version, crates = workspace_crates(repo)
    roots = [crate.name for crate in crates if not crate.internal_deps]
    dependents = [crate for crate in crates if crate.internal_deps]
    require(roots, "publish plan must include at least one root crate")
    if args.package:
        package_workspace(repo, version)
    if args.dry_run:
        dry_run_roots(repo, crates)
    print(
        f"validated publish plan for {len(crates)} crate(s) at {version}; "
        f"dry-run roots={','.join(roots)}; "
        f"internal-dependent={','.join(crate.name for crate in dependents) or 'none'}"
    )


if __name__ == "__main__":
    main()
