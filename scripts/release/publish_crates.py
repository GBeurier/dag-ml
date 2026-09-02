#!/usr/bin/env python3
"""Publish workspace crates to crates.io in dependency order."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn


ALREADY_UPLOADED = re.compile(
    r"already (exists|uploaded)|is already being published|crate version .* is already",
    re.IGNORECASE,
)
SHA1 = re.compile(r"[0-9a-f]{40}")
SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_REGISTRY_METADATA_BYTES = 1 << 20
MAX_CRATE_BYTES = 64 << 20
MAX_CRATE_MEMBERS = 10_000
MAX_VCS_INFO_BYTES = 64 << 10


@dataclass(frozen=True)
class Crate:
    name: str
    manifest: Path
    internal_deps: tuple[str, ...]


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def repository_head(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    require(proc.returncode == 0, f"cannot resolve release HEAD: {proc.stderr.strip()}")
    head = proc.stdout.strip()
    require(
        bool(SHA1.fullmatch(head)),
        f"release HEAD is not a full lowercase SHA-1: {head}",
    )
    return head


def _read_bounded(response: Any, limit: int, label: str) -> bytes:
    payload = response.read(limit + 1)
    require(len(payload) <= limit, f"{label} exceeds the {limit}-byte limit")
    return payload


def _strict_json_object(payload: bytes, label: str) -> dict[str, Any]:
    def object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            payload.decode("utf-8"), object_pairs_hook=object_no_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"cannot decode {label}: {error}")
    require(isinstance(parsed, dict), f"{label} must be a JSON object")
    return parsed


def verify_crate_archive(
    name: str,
    version: str,
    expected_head: str,
    registry_checksum: str,
    crate_bytes: bytes,
) -> None:
    """Fail closed unless a registry crate is exact and clean at ``expected_head``."""

    require(
        bool(SHA256.fullmatch(registry_checksum)),
        f"crates.io returned an invalid checksum for {name} {version}",
    )
    actual_checksum = hashlib.sha256(crate_bytes).hexdigest()
    require(
        actual_checksum == registry_checksum,
        f"downloaded {name} {version} checksum {actual_checksum} does not match "
        f"crates.io {registry_checksum}",
    )
    require(
        bool(SHA1.fullmatch(expected_head)),
        f"expected release commit is not a full lowercase SHA-1: {expected_head}",
    )

    try:
        with tarfile.open(fileobj=io.BytesIO(crate_bytes), mode="r:gz") as archive:
            members = archive.getmembers()
            require(
                len(members) <= MAX_CRATE_MEMBERS,
                f"{name} {version} crate has too many archive members",
            )
            vcs_members = [
                member
                for member in members
                if PurePosixPath(member.name).name == ".cargo_vcs_info.json"
            ]
            require(
                len(vcs_members) == 1,
                f"{name} {version} crate must contain exactly one .cargo_vcs_info.json",
            )
            vcs_member = vcs_members[0]
            expected_path = f"{name}-{version}/.cargo_vcs_info.json"
            vcs_path = PurePosixPath(vcs_member.name)
            require(
                "\\" not in vcs_member.name
                and not vcs_path.is_absolute()
                and all(part not in {"", ".", ".."} for part in vcs_path.parts)
                and vcs_member.name == expected_path,
                f"{name} {version} crate has an unsafe or misplaced VCS record: "
                f"{vcs_member.name}",
            )
            require(
                vcs_member.isfile(),
                f"{name} {version} VCS record must be a regular archive member",
            )
            require(
                0 < vcs_member.size <= MAX_VCS_INFO_BYTES,
                f"{name} {version} VCS record has an invalid size",
            )
            extracted = archive.extractfile(vcs_member)
            if extracted is None:
                fail(f"cannot read {name} {version} VCS record")
            vcs_bytes = extracted.read(MAX_VCS_INFO_BYTES + 1)
            require(
                len(vcs_bytes) == vcs_member.size <= MAX_VCS_INFO_BYTES,
                f"{name} {version} VCS record size changed while reading",
            )
    except (tarfile.TarError, OSError) as error:
        fail(f"cannot inspect downloaded {name} {version} crate: {error}")

    vcs = _strict_json_object(vcs_bytes, f"{name} {version} .cargo_vcs_info.json")
    git = vcs.get("git")
    if not isinstance(git, dict):
        fail(f"{name} {version} VCS record has no git object")
    require(
        git.get("dirty") is False,
        f"{name} {version} registry crate is not explicitly attested clean",
    )
    sha1 = git.get("sha1")
    if not isinstance(sha1, str) or SHA1.fullmatch(sha1) is None:
        fail(f"{name} {version} VCS record has no full lowercase git sha1")
    require(
        sha1 == expected_head,
        f"{name} {version} registry crate commit {sha1} does not match HEAD {expected_head}",
    )


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


def package_version(
    package: dict[str, Any], workspace_version: str, manifest_path: Path
) -> str:
    value = package.get("version")
    if isinstance(value, dict):
        require(
            value.get("workspace") is True,
            f"{manifest_path}: package.version table must inherit from workspace",
        )
        return workspace_version
    if not isinstance(value, str):
        fail(f"{manifest_path}: package.version is missing")
    return value


def topo_sort(crates: list[Crate]) -> list[Crate]:
    remaining = {crate.name: crate for crate in crates}
    ordered: list[Crate] = []
    published: set[str] = set()

    while remaining:
        ready = sorted(
            [
                crate
                for crate in remaining.values()
                if set(crate.internal_deps).issubset(published)
            ],
            key=lambda crate: crate.name,
        )
        require(
            bool(ready),
            "publish plan contains an internal dependency cycle: "
            + ", ".join(sorted(remaining)),
        )
        for crate in ready:
            ordered.append(crate)
            published.add(crate.name)
            del remaining[crate.name]

    return ordered


def workspace_crates(repo: Path) -> tuple[str, list[Crate]]:
    root = load_toml(repo / "Cargo.toml")
    workspace = root["workspace"]
    workspace_version = workspace["package"]["version"]
    workspace_deps = workspace.get("dependencies", {})

    manifests: dict[str, dict[str, Any]] = {}
    manifest_paths: dict[str, Path] = {}

    for member in workspace["members"]:
        manifest_path = repo / member / "Cargo.toml"
        manifest = load_toml(manifest_path)
        package = manifest["package"]
        if package.get("publish") is False:
            continue
        name = package["name"]
        version = package_version(package, workspace_version, manifest_path)
        require(
            version == workspace_version,
            f"{manifest_path}: package.version must equal workspace version {workspace_version}",
        )
        manifests[name] = manifest
        manifest_paths[name] = manifest_path

    crates: list[Crate] = []
    package_names = set(manifests)
    for name, manifest in manifests.items():
        internal = sorted(dependency_names(manifest).intersection(package_names))
        for dep_name in internal:
            dep = workspace_deps.get(dep_name)
            require(
                isinstance(dep, dict),
                f"workspace dependency {dep_name} must be a table",
            )
            require(
                dep.get("path"), f"workspace dependency {dep_name} must declare path"
            )
            require(
                dep.get("version") == workspace_version,
                f"workspace dependency {dep_name} must pin version {workspace_version}",
            )
        crates.append(
            Crate(
                name=name,
                manifest=manifest_paths[name],
                internal_deps=tuple(internal),
            )
        )

    require(bool(crates), "publish plan has no publishable workspace crates")
    return workspace_version, topo_sort(crates)


def validate_tag(tag: str, version: str) -> None:
    require(tag.startswith("v"), f"release tag must start with v: {tag}")
    tag_version = tag.removeprefix("v")
    require(
        tag_version == version,
        f"release tag {tag} does not match workspace version {version}",
    )


def cargo_publish(crate: Crate, dry_run: bool, no_verify: bool) -> str:
    cmd = ["cargo", "publish", "-p", crate.name]
    if dry_run:
        cmd.extend(["--dry-run", "--allow-dirty"])
    if no_verify:
        cmd.append("--no-verify")

    print(f"::group::publish {crate.name} (dry_run={int(dry_run)})", flush=True)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    print("::endgroup::", flush=True)

    if proc.returncode == 0:
        return "published"
    if not dry_run and ALREADY_UPLOADED.search(proc.stdout or ""):
        print(f"::notice::{crate.name} version already exists on crates.io; continuing")
        return "already"
    raise SystemExit(proc.returncode)


def crate_version_exists(name: str, version: str, expected_head: str) -> bool:
    request = urllib.request.Request(
        f"https://crates.io/api/v1/crates/{name}/{version}",
        headers={"User-Agent": "dag-ml-release-script"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata_bytes = _read_bounded(
                response,
                MAX_REGISTRY_METADATA_BYTES,
                f"crates.io metadata for {name} {version}",
            )
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise

    metadata = _strict_json_object(
        metadata_bytes, f"crates.io metadata for {name} {version}"
    )
    registry_version = metadata.get("version")
    if not isinstance(registry_version, dict):
        fail(f"crates.io metadata for {name} {version} has no version object")
    require(
        registry_version.get("crate") == name
        and registry_version.get("num") == version,
        f"crates.io returned mismatched identity for {name} {version}",
    )
    checksum = registry_version.get("checksum")
    if not isinstance(checksum, str):
        fail(f"crates.io metadata for {name} {version} has no checksum")
    expected_download_path = f"/api/v1/crates/{name}/{version}/download"
    require(
        registry_version.get("dl_path") == expected_download_path,
        f"crates.io returned an unexpected download path for {name} {version}",
    )
    download = urllib.request.Request(
        "https://crates.io" + expected_download_path,
        headers={"User-Agent": "dag-ml-release-script"},
    )
    with urllib.request.urlopen(download, timeout=30) as response:
        crate_bytes = _read_bounded(
            response, MAX_CRATE_BYTES, f"downloaded {name} {version} crate"
        )
    verify_crate_archive(name, version, expected_head, checksum, crate_bytes)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="verify independent publish roots; same-release dependents require an indexed root",
    )
    parser.add_argument("--tag", help="release tag to validate, for example v0.2.0")
    parser.add_argument(
        "--no-verify", action="store_true", help="pass --no-verify to cargo publish"
    )
    parser.add_argument(
        "--plan-only", action="store_true", help="print the publish order and exit"
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=120,
        help="delay after each successful upload so the crates.io sparse index catches up",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    version, crates = workspace_crates(repo)
    head = repository_head(repo)
    if args.tag:
        validate_tag(args.tag, version)

    print(
        f"publish plan for {len(crates)} crate(s) at {version}: "
        + " -> ".join(crate.name for crate in crates)
    )
    if args.plan_only:
        return

    if not args.dry_run and "CARGO_REGISTRY_TOKEN" not in os.environ:
        fail("CARGO_REGISTRY_TOKEN is required for cargo publish authentication")

    for index, crate in enumerate(crates):
        # `cargo publish --dry-run` resolves dependencies through crates.io.
        # A dependent workspace crate therefore cannot be dry-run until the
        # root crate from this *same* release has actually been indexed.  The
        # release-plan validator has already checked the whole dependency DAG;
        # exercise every independent root here and leave the dependent crates
        # to the real, topologically ordered tagged release.
        if args.dry_run and crate.internal_deps:
            print(
                "::notice::skipping dry-run for "
                f"{crate.name}: awaits same-release internal dependency "
                + ", ".join(crate.internal_deps)
            )
            continue
        if not args.dry_run and crate_version_exists(crate.name, version, head):
            print(
                f"::notice::{crate.name} {version} already exists on crates.io; skipping"
            )
            continue
        result = cargo_publish(crate, dry_run=args.dry_run, no_verify=args.no_verify)
        if not args.dry_run and result == "already":
            require(
                crate_version_exists(crate.name, version, head),
                f"cargo reported {crate.name} {version} already uploaded but crates.io "
                "does not expose a verifiable artifact",
            )
        if (
            not args.dry_run
            and result == "published"
            and args.sleep_seconds > 0
            and index < len(crates) - 1
        ):
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
