from __future__ import annotations

import os
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the current workspace.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "test_methods_optimizer_local.sh"
METHODS_PIN = "0ef355e6f74573ed07a6920bdeed1a052a6e8312"


def _scaffold_methods_checkout(path: Path, *, initialize_git: bool) -> None:
    binding = path / "bindings" / "rust" / "n4m"
    binding.mkdir(parents=True)
    (binding / "Cargo.toml").write_text(
        '[package]\nname = "n4m"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    if initialize_git:
        subprocess.run(["git", "init", "-q"], cwd=path, check=True)
        subprocess.run(["git", "config", "user.name", "pin-test"], cwd=path, check=True)
        subprocess.run(
            ["git", "config", "user.email", "pin-test@example.invalid"],
            cwd=path,
            check=True,
        )
        subprocess.run(["git", "add", "-A"], cwd=path, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=path, check=True)


def _probe(checkout: Path, sha: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update({"N4M_METHODS_REPO": str(checkout), "N4M_BINDING_SHA": sha})
    return subprocess.run(
        [str(SCRIPT), "--probe"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_probe_refuses_malformed_binding_sha(tmp_path: Path) -> None:
    checkout = tmp_path / "methods"
    _scaffold_methods_checkout(checkout, initialize_git=False)

    result = _probe(checkout, "main")

    assert result.returncode == 2
    assert "full 40-character commit SHA" in result.stderr


def test_probe_refuses_checkout_at_different_sha(tmp_path: Path) -> None:
    checkout = tmp_path / "methods"
    _scaffold_methods_checkout(checkout, initialize_git=True)

    result = _probe(checkout, METHODS_PIN)

    assert result.returncode == 2
    assert f"Methods binding SHA mismatch: expected {METHODS_PIN}" in result.stderr


def test_ci_and_overlay_use_the_reviewed_methods_pin() -> None:
    overlay = tomllib.loads(
        (ROOT / "crates" / "dag-ml-core" / "Cargo.toml.methods-local").read_text(
            encoding="utf-8"
        )
    )
    dependency = overlay["dependencies"]["n4m"]
    assert dependency["git"] == "https://github.com/GBeurier/nirs4all-methods.git"
    assert dependency["rev"] == METHODS_PIN
    assert "path" not in dependency

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    methods_job = workflow.split("  methods-hpo-local:", maxsplit=1)[1].split(
        "\n  sanitizer:", maxsplit=1
    )[0]
    assert f"ref: {METHODS_PIN}" in methods_job
    assert f"N4M_BINDING_SHA: {METHODS_PIN}" in methods_job
