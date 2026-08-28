"""Regression coverage for the DAG-ML crates.io release driver."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "release" / "publish_crates.py"
SPEC = importlib.util.spec_from_file_location("publish_crates", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publish_crates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = publish_crates
SPEC.loader.exec_module(publish_crates)


def _plan() -> list[object]:
    return [
        publish_crates.Crate("dag-ml-core", ROOT / "core.toml", ()),
        publish_crates.Crate("dag-ml", ROOT / "dag-ml.toml", ("dag-ml-core",)),
        publish_crates.Crate("dag-ml-cli", ROOT / "cli.toml", ("dag-ml-core",)),
    ]


def test_dry_run_checks_only_publishable_roots(
    monkeypatch, capsys
) -> None:
    """A dry-run cannot resolve crates which this release has not indexed yet."""
    calls: list[tuple[str, bool, bool]] = []
    monkeypatch.setattr(publish_crates, "workspace_crates", lambda _: ("0.3.19", _plan()))
    monkeypatch.setattr(
        publish_crates,
        "cargo_publish",
        lambda crate, dry_run, no_verify: calls.append((crate.name, dry_run, no_verify))
        or "published",
    )
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


def test_real_publish_keeps_topological_order(monkeypatch) -> None:
    """The tagged publication path must still upload every crate in order."""
    calls: list[tuple[str, bool, bool]] = []
    monkeypatch.setenv("CARGO_REGISTRY_TOKEN", "test-token")
    monkeypatch.setattr(publish_crates, "workspace_crates", lambda _: ("0.3.19", _plan()))
    monkeypatch.setattr(
        publish_crates,
        "cargo_publish",
        lambda crate, dry_run, no_verify: calls.append((crate.name, dry_run, no_verify))
        or "published",
    )
    monkeypatch.setattr(publish_crates, "crate_version_exists", lambda *_: False)
    monkeypatch.setattr(sys, "argv", ["publish_crates.py", "--sleep-seconds", "0"])

    publish_crates.main()

    assert calls == [
        ("dag-ml-core", False, False),
        ("dag-ml", False, False),
        ("dag-ml-cli", False, False),
    ]
