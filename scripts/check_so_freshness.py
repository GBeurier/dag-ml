#!/usr/bin/env python3
"""Guard against a stale committed extension binary (the tracked ``_dag_ml.abi3.so``).

A real bug shipped when the git-tracked ``crates/dag-ml-py/python/dag_ml/_dag_ml.abi3.so``
had been built at an old commit while the committed Rust sources kept advancing, so the
committed binary silently ran old numerics. This check fails the green gate when the
tracked ``.so``'s last commit predates the newest commit that touched the Rust sources
that compile into it.

mtime is unreliable across clones, so the authoritative signal is the git-commit-touch
time (``git log -1 --format=%ct -- <path>``): the binary's last-touch commit time vs the
max last-touch commit time over the Rust tree (core + py crate sources and their Cargo
manifests / lockfile). The mtime is reported only as informational context.

Exit codes:
  0  fresh, OR skipped gracefully (not a git repo / missing .so / no commit history)
  1  stale: the tracked .so was last committed before newer Rust sources

Run ``python3 scripts/check_so_freshness.py --self-test`` to exercise both branches.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Tracked extension binary, relative to the repo root.
SO_RELATIVE = "crates/dag-ml-py/python/dag_ml/_dag_ml.abi3.so"

# Rust sources that compile into the .so: the py crate, its only path dependency
# (dag-ml-core), their Cargo manifests, and the workspace manifest + lockfile.
RUST_DIRS = (
    "crates/dag-ml-core/src",
    "crates/dag-ml-py/src",
)
RUST_FILES = (
    "crates/dag-ml-core/Cargo.toml",
    "crates/dag-ml-py/Cargo.toml",
    "Cargo.toml",
    "Cargo.lock",
)
RUST_SUFFIX = ".rs"

NOTICE = "check_so_freshness:"


def is_git_repo(repo: Path) -> bool:
    """Return True if ``repo`` is inside a git work tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def last_commit_ts(repo: Path, relative: str) -> int | None:
    """Return the unix timestamp of the last commit that touched ``relative``.

    Returns None when the path has no commit history (untracked / never committed).
    """
    result = subprocess.run(
        ["git", "log", "-1", "--format=%ct", "--", relative],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    return int(text)


def rust_paths(repo: Path) -> list[str]:
    """Collect the Rust-source paths (relative, posix) that feed the binary."""
    paths: list[str] = []
    for directory in RUST_DIRS:
        root = repo / directory
        if not root.exists():
            continue
        for path in sorted(root.rglob(f"*{RUST_SUFFIX}")):
            if any(part in {"target", "__pycache__"} for part in path.parts):
                continue
            paths.append(path.relative_to(repo).as_posix())
    for relative in RUST_FILES:
        if (repo / relative).exists():
            paths.append(relative)
    return paths


def newest_rust_commit(repo: Path, paths: list[str]) -> tuple[int, str] | None:
    """Return (timestamp, path) of the most recently committed Rust source."""
    best: tuple[int, str] | None = None
    for relative in paths:
        ts = last_commit_ts(repo, relative)
        if ts is None:
            continue
        if best is None or ts > best[0]:
            best = (ts, relative)
    return best


def check(repo: Path) -> int:
    """Run the freshness check against ``repo``; return a process exit code."""
    so_path = repo / SO_RELATIVE
    if not so_path.exists():
        print(f"{NOTICE} skip — extension binary {SO_RELATIVE} not present.")
        return 0
    if not is_git_repo(repo):
        print(f"{NOTICE} skip — {repo} is not a git work tree.")
        return 0

    so_ts = last_commit_ts(repo, SO_RELATIVE)
    if so_ts is None:
        print(f"{NOTICE} skip — {SO_RELATIVE} has no commit history (untracked?).")
        return 0

    paths = rust_paths(repo)
    newest = newest_rust_commit(repo, paths)
    if newest is None:
        print(f"{NOTICE} skip — no committed Rust sources found to compare against.")
        return 0

    rust_ts, rust_path = newest
    so_mtime = int(so_path.stat().st_mtime)
    if so_ts < rust_ts:
        newer = [p for p in paths if (ts := last_commit_ts(repo, p)) is not None and ts > so_ts]
        listing = "\n".join(f"  - {p}" for p in newer)
        print(
            f"{NOTICE} STALE — the committed extension binary is older than its Rust sources.\n"
            f"  tracked .so : {SO_RELATIVE} (last commit ct={so_ts}, mtime={so_mtime})\n"
            f"  newest Rust : {rust_path} (last commit ct={rust_ts})\n"
            f"  Rust files committed after the .so:\n{listing}\n"
            "  Remediation: rebuild + commit the .so via "
            "`maturin develop --release` (in crates/dag-ml-py), then `git add` the .so.",
            file=sys.stderr,
        )
        return 1

    print(
        f"{NOTICE} fresh — {SO_RELATIVE} (ct={so_ts}) is at/after the newest Rust source "
        f"{rust_path} (ct={rust_ts}); checked {len(paths)} Rust path(s)."
    )
    return 0


def self_test() -> int:
    """Exercise the comparison logic on a synthetic git repo (fresh + stale)."""
    import os
    import tempfile

    def git(repo: Path, *args: str, ts: int | None = None) -> None:
        env = dict(os.environ)
        env["GIT_AUTHOR_NAME"] = env["GIT_COMMITTER_NAME"] = "selftest"
        env["GIT_AUTHOR_EMAIL"] = env["GIT_COMMITTER_EMAIL"] = "selftest@example.com"
        if ts is not None:
            stamp = f"@{ts} +0000"
            env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = stamp
        subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)

    def scaffold(repo: Path) -> None:
        (repo / "crates/dag-ml-core/src").mkdir(parents=True)
        (repo / "crates/dag-ml-py/src").mkdir(parents=True)
        (repo / "crates/dag-ml-py/python/dag_ml").mkdir(parents=True)
        (repo / "crates/dag-ml-core/src/lib.rs").write_text("// core\n", encoding="utf-8")
        (repo / "crates/dag-ml-py/src/lib.rs").write_text("// py\n", encoding="utf-8")
        for name in ("crates/dag-ml-core/Cargo.toml", "crates/dag-ml-py/Cargo.toml", "Cargo.toml"):
            (repo / name).write_text("# cargo\n", encoding="utf-8")
        (repo / "Cargo.lock").write_text("# lock\n", encoding="utf-8")
        (repo / SO_RELATIVE).write_bytes(b"\x7fELF binary")
        git(repo, "init", "-q")

    failures: list[str] = []

    # Case 1: FRESH — .so committed last, so its commit time wins.
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        scaffold(repo)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "rust", ts=1_000_000)
        # Advance only the .so in a later commit.
        (repo / SO_RELATIVE).write_bytes(b"\x7fELF rebuilt")
        git(repo, "add", SO_RELATIVE)
        git(repo, "commit", "-q", "-m", "rebuild so", ts=2_000_000)
        code = check(repo)
        if code != 0:
            failures.append(f"FRESH case expected exit 0, got {code}")

    # Case 2: STALE — Rust advances after the .so's last commit.
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        scaffold(repo)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "initial with so", ts=1_000_000)
        # Advance Rust source in a later commit; the .so stays at the old commit.
        (repo / "crates/dag-ml-core/src/lib.rs").write_text("// core v2\n", encoding="utf-8")
        git(repo, "add", "crates/dag-ml-core/src/lib.rs")
        git(repo, "commit", "-q", "-m", "advance rust", ts=2_000_000)
        code = check(repo)
        if code != 1:
            failures.append(f"STALE case expected exit 1, got {code}")

    if failures:
        for line in failures:
            print(f"{NOTICE} self-test FAILED: {line}", file=sys.stderr)
        return 1
    print(f"{NOTICE} self-test passed (fresh -> exit 0, stale -> exit 1).")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        return self_test()
    repo = Path(__file__).resolve().parents[1]
    return check(repo)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
