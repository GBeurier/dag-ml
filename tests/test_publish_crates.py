"""Regression coverage for the DAG-ML crates.io release driver."""

from __future__ import annotations

import hashlib
import io
import importlib.util
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release" / "publish_crates.py"
SPEC = importlib.util.spec_from_file_location("publish_crates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publish_crates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publish_crates
SPEC.loader.exec_module(publish_crates)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _crate_bytes(
    *,
    name: str = "dag-ml-core",
    version: str = "0.3.23",
    head: str = "1" * 40,
    vcs_path: str | None = None,
    dirty: bool | None = False,
    duplicate_vcs: bool = False,
    vcs_symlink: bool = False,
) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        readme = b"published crate\n"
        readme_info = tarfile.TarInfo(f"{name}-{version}/README.md")
        readme_info.size = len(readme)
        archive.addfile(readme_info, io.BytesIO(readme))

        if vcs_path is not None:
            vcs: dict[str, Any] = {"git": {"sha1": head}}
            if dirty is not None:
                vcs["git"]["dirty"] = dirty
            vcs_bytes = json.dumps(vcs, sort_keys=True).encode("utf-8")
            vcs_info = tarfile.TarInfo(vcs_path)
            if vcs_symlink:
                vcs_info.type = tarfile.SYMTYPE
                vcs_info.linkname = f"{name}-{version}/README.md"
            else:
                vcs_info.size = len(vcs_bytes)
            archive.addfile(vcs_info, None if vcs_symlink else io.BytesIO(vcs_bytes))
            if duplicate_vcs:
                duplicate = tarfile.TarInfo(
                    f"{name}-{version}/nested/.cargo_vcs_info.json"
                )
                duplicate.size = len(vcs_bytes)
                archive.addfile(duplicate, io.BytesIO(vcs_bytes))
    return buffer.getvalue()


def _verify(payload: bytes, *, head: str = "1" * 40) -> None:
    publish_crates.verify_crate_archive(
        "dag-ml-core",
        "0.3.23",
        head,
        hashlib.sha256(payload).hexdigest(),
        payload,
    )


def _plan() -> list[Any]:
    return [
        publish_crates.Crate("dag-ml-core", ROOT / "core.toml", ()),
        publish_crates.Crate("dag-ml", ROOT / "dag-ml.toml", ("dag-ml-core",)),
        publish_crates.Crate("dag-ml-cli", ROOT / "cli.toml", ("dag-ml-core",)),
    ]


def test_dry_run_checks_only_publishable_roots(monkeypatch, capsys) -> None:
    """A dry-run cannot resolve crates which this release has not indexed yet."""
    calls: list[tuple[str, bool, bool]] = []

    def record_publish(crate: Any, dry_run: bool, no_verify: bool) -> str:
        calls.append((crate.name, dry_run, no_verify))
        return "published"

    monkeypatch.setattr(
        publish_crates, "workspace_crates", lambda _: ("0.3.19", _plan())
    )
    monkeypatch.setattr(publish_crates, "cargo_publish", record_publish)
    monkeypatch.setattr(
        publish_crates,
        "crate_version_exists",
        lambda *_: (_ for _ in ()).throw(AssertionError("network lookup in dry-run")),
    )
    monkeypatch.setattr(sys, "argv", ["publish_crates.py", "--dry-run"])

    publish_crates.main()

    assert calls == [("dag-ml-core", True, False)]
    output = capsys.readouterr().out
    assert "skipping dry-run for dag-ml" in output
    assert "skipping dry-run for dag-ml-cli" in output


def test_cargo_dry_run_is_forced_offline(monkeypatch) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **_: Any) -> Any:
        commands.append(command)
        return publish_crates.subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(publish_crates.subprocess, "run", run)

    result = publish_crates.cargo_publish(_plan()[0], dry_run=True, no_verify=False)

    assert result == "published"
    assert commands == [
        [
            "cargo",
            "publish",
            "-p",
            "dag-ml-core",
            "--dry-run",
            "--allow-dirty",
            "--offline",
        ]
    ]


def test_real_publish_keeps_topological_order(monkeypatch) -> None:
    """The tagged publication path must still upload every crate in order."""
    calls: list[tuple[str, bool, bool]] = []

    def record_publish(crate: Any, dry_run: bool, no_verify: bool) -> str:
        calls.append((crate.name, dry_run, no_verify))
        return "published"

    monkeypatch.setenv("CARGO_REGISTRY_TOKEN", "test-token")
    monkeypatch.setattr(
        publish_crates, "workspace_crates", lambda _: ("0.3.19", _plan())
    )
    monkeypatch.setattr(publish_crates, "cargo_publish", record_publish)
    monkeypatch.setattr(publish_crates, "crate_version_exists", lambda *_: False)
    monkeypatch.setattr(sys, "argv", ["publish_crates.py", "--sleep-seconds", "0"])

    publish_crates.main()

    assert calls == [
        ("dag-ml-core", False, False),
        ("dag-ml", False, False),
        ("dag-ml-cli", False, False),
    ]


def test_existing_version_download_is_checksum_and_vcs_qualified(monkeypatch) -> None:
    head = "1" * 40
    payload = _crate_bytes(
        head=head,
        vcs_path="dag-ml-core-0.3.23/.cargo_vcs_info.json",
    )
    metadata = {
        "version": {
            "crate": "dag-ml-core",
            "num": "0.3.23",
            "checksum": hashlib.sha256(payload).hexdigest(),
            "dl_path": "/api/v1/crates/dag-ml-core/0.3.23/download",
        }
    }
    responses = iter(
        [_Response(json.dumps(metadata).encode("utf-8")), _Response(payload)]
    )
    monkeypatch.setattr(
        publish_crates.urllib.request, "urlopen", lambda *_a, **_k: next(responses)
    )

    assert publish_crates.crate_version_exists("dag-ml-core", "0.3.23", head)


def test_real_publish_skips_only_after_existing_artifact_is_qualified(
    monkeypatch, capsys
) -> None:
    checks: list[tuple[str, str, str]] = []
    monkeypatch.setenv("CARGO_REGISTRY_TOKEN", "test-token")
    monkeypatch.setattr(
        publish_crates,
        "workspace_crates",
        lambda _: ("0.3.23", [_plan()[0]]),
    )
    monkeypatch.setattr(publish_crates, "repository_head", lambda _: "1" * 40)

    def qualified(name: str, version: str, head: str) -> bool:
        checks.append((name, version, head))
        return True

    monkeypatch.setattr(publish_crates, "crate_version_exists", qualified)
    monkeypatch.setattr(
        publish_crates,
        "cargo_publish",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("cargo publish ran before an attested skip")
        ),
    )
    monkeypatch.setattr(sys, "argv", ["publish_crates.py", "--sleep-seconds", "0"])

    publish_crates.main()

    assert checks == [("dag-ml-core", "0.3.23", "1" * 40)]
    assert "already exists on crates.io; skipping" in capsys.readouterr().out


def test_registry_checksum_mismatch_is_fatal() -> None:
    payload = _crate_bytes(
        vcs_path="dag-ml-core-0.3.23/.cargo_vcs_info.json",
    )
    with pytest.raises(SystemExit, match="does not match crates.io"):
        publish_crates.verify_crate_archive(
            "dag-ml-core", "0.3.23", "1" * 40, "0" * 64, payload
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vcs_path": None}, "exactly one"),
        (
            {
                "vcs_path": "dag-ml-core-0.3.23/.cargo_vcs_info.json",
                "duplicate_vcs": True,
            },
            "exactly one",
        ),
        ({"vcs_path": "../.cargo_vcs_info.json"}, "unsafe or misplaced"),
        (
            {
                "vcs_path": "dag-ml-core-0.3.23/.cargo_vcs_info.json",
                "vcs_symlink": True,
            },
            "regular archive member",
        ),
        (
            {
                "vcs_path": "dag-ml-core-0.3.23/.cargo_vcs_info.json",
                "dirty": True,
            },
            "not explicitly attested clean",
        ),
        (
            {
                "vcs_path": "dag-ml-core-0.3.23/.cargo_vcs_info.json",
                "dirty": None,
            },
            "not explicitly attested clean",
        ),
    ],
)
def test_vcs_record_attacks_are_rejected(kwargs: dict[str, Any], message: str) -> None:
    payload = _crate_bytes(**kwargs)
    with pytest.raises(SystemExit, match=message):
        _verify(payload)


def test_registry_vcs_commit_substitution_is_fatal() -> None:
    payload = _crate_bytes(
        head="2" * 40,
        vcs_path="dag-ml-core-0.3.23/.cargo_vcs_info.json",
    )
    with pytest.raises(SystemExit, match="does not match HEAD"):
        _verify(payload, head="1" * 40)


def test_cargo_already_race_is_requalified_before_continue(monkeypatch) -> None:
    checks: list[tuple[str, str, str]] = []
    monkeypatch.setenv("CARGO_REGISTRY_TOKEN", "test-token")
    monkeypatch.setattr(
        publish_crates,
        "workspace_crates",
        lambda _: ("0.3.23", [_plan()[0]]),
    )
    monkeypatch.setattr(publish_crates, "repository_head", lambda _: "1" * 40)
    monkeypatch.setattr(publish_crates, "cargo_publish", lambda *_a, **_k: "already")

    def exists(name: str, version: str, head: str) -> bool:
        checks.append((name, version, head))
        return len(checks) == 2

    monkeypatch.setattr(publish_crates, "crate_version_exists", exists)
    monkeypatch.setattr(sys, "argv", ["publish_crates.py", "--sleep-seconds", "0"])

    publish_crates.main()

    assert checks == [
        ("dag-ml-core", "0.3.23", "1" * 40),
        ("dag-ml-core", "0.3.23", "1" * 40),
    ]
