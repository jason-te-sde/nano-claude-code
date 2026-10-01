# src/nanoclaude/tools/search.py
"""Finding files and finding text in them.

ripgrep is the backend when it is on PATH: it is faster than anything written
here and it already understands .gitignore. When it is missing the pure-Python
fallback does the same job slowly, because "install ripgrep first" is not an
acceptable thing to say to someone who just ran pipx install.

The two backends are covered by the same test so the fallback cannot quietly
become a different tool with the same name.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pathspec
from pathspec.pattern import Pattern as PathspecPattern

from nanoclaude.permissions.rules import GLOB_PATTERN_FACTORY

BINARY_SNIFF_BYTES = 8192
DEFAULT_LIMIT = 200


@dataclass(frozen=True, slots=True)
class Match:
    path: str
    line_no: int
    line: str


def ripgrep_available() -> bool:
    return shutil.which("rg") is not None


def _ignore_spec(root: str) -> pathspec.PathSpec[PathspecPattern]:
    lines = ["/.git/"]
    gitignore = Path(root) / ".gitignore"
    if gitignore.exists():
        lines.extend(gitignore.read_text(errors="replace").splitlines())
    return pathspec.PathSpec.from_lines(GLOB_PATTERN_FACTORY, lines)


def walk_files(root: str, *, pattern: str | None = None, limit: int = DEFAULT_LIMIT) -> list[str]:
    """Absolute paths under ``root``, gitignore-aware, most recently modified first."""
    ignore = _ignore_spec(root)
    glob_spec = pathspec.PathSpec.from_lines(GLOB_PATTERN_FACTORY, [pattern]) if pattern else None
    found: list[tuple[float, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        relative_dir = os.path.relpath(dirpath, root)
        dirnames[:] = [
            d
            for d in dirnames
            # os.path.join, not Path.joinpath: the trailing "" component is
            # what makes this a directory-shaped candidate ("sub/dir/") for
            # pathspec's trailing-slash-sensitive directory patterns -- a
            # Path-built equivalent would need the same string suffix glued
            # back on anyway, for no gain in clarity.
            if not ignore.match_file(os.path.join(relative_dir, d, "").lstrip("./"))  # noqa: PTH118
        ]
        for name in filenames:
            absolute = os.path.join(dirpath, name)  # noqa: PTH118
            relative = os.path.relpath(absolute, root)
            if ignore.match_file(relative):
                continue
            if glob_spec is not None and not glob_spec.match_file(relative):
                continue
            try:
                found.append((os.path.getmtime(absolute), absolute))  # noqa: PTH204
            except OSError:
                continue
    found.sort(reverse=True)
    return [path for _, path in found[:limit]]


def ripgrep_search(
    pattern: str,
    *,
    root: str,
    glob: str | None,
    ignore_case: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> list[Match]:
    # --hidden and the /.git/ exclusion bring this in line with walk_files,
    # which already includes dotfiles (Grep's own secret-file filter depends
    # on an un-ignored .env actually being found) and always excludes .git.
    # Without --hidden, ripgrep skips every dotfile by default, silently
    # disagreeing with the fallback on exactly the files the secret-file
    # filter exists to catch.
    argv = [
        "rg",
        "--json",
        "--line-number",
        "--no-heading",
        "--hidden",
        "--glob",
        "!/.git/",
    ]
    if ignore_case:
        argv.append("-i")
    if glob:
        argv += ["--glob", glob]
    argv += ["--", pattern, root]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.strip() or "ripgrep failed")
    matches: list[Match] = []
    for line in completed.stdout.splitlines():
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        data = event["data"]
        matches.append(
            Match(
                data["path"]["text"],
                data["line_number"],
                data["lines"]["text"].rstrip("\n"),
            )
        )
        if len(matches) >= limit:
            break
    return matches


def python_search(
    pattern: str,
    *,
    root: str,
    glob: str | None,
    ignore_case: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> list[Match]:
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    matches: list[Match] = []
    for path in sorted(walk_files(root, pattern=glob and f"**/{glob}", limit=10_000)):
        try:
            data = Path(path).read_bytes()
        except OSError:
            continue
        if b"\0" in data[:BINARY_SNIFF_BYTES]:
            continue
        for line_no, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if regex.search(line):
                matches.append(Match(path, line_no, line))
                if len(matches) >= limit:
                    return matches
    return matches


def search(
    pattern: str,
    *,
    root: str,
    glob: str | None = None,
    ignore_case: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> list[Match]:
    backend = ripgrep_search if ripgrep_available() else python_search
    return backend(pattern, root=root, glob=glob, ignore_case=ignore_case, limit=limit)
