# src/nanoclaude/tools/search.py
"""Finding files and finding text in them.

ripgrep is the backend when it is on PATH: it is faster than anything written
here and it already understands .gitignore. When it is missing the pure-Python
fallback does the same job slowly, because "install ripgrep first" is not an
acceptable thing to say to someone who just ran pipx install.

The two backends are covered by the same test so the fallback cannot quietly
become a different tool with the same name. That includes links: ripgrep does not
follow a symbolic link, to a file or to a directory, so a file is searched under the
name it really has and never through a link to it. The fallback does the same.
Following links is how a link inside the project to a file outside it would have its
lines returned.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pathspec
from pathspec.pattern import Pattern as PathspecPattern

from nanoclaude.permissions.rules import GLOB_PATTERN_FACTORY

BINARY_SNIFF_BYTES = 8192
#: Files larger than this are not searched, by either backend. The fallback
#: reads each file whole, so without a cap one large data file would make a
#: search slow and memory-hungry; ripgrep is given the same limit so the two
#: backends keep answering identically.
MAX_SEARCH_BYTES = 10 * 1024 * 1024
DEFAULT_LIMIT = 200


@dataclass(frozen=True, slots=True)
class Match:
    path: str
    line_no: int
    line: str
    #: True for a context line (-A/-B/-C) shown around a match, not itself a
    #: match. Defaulted so every existing positional Match(path, line_no, line)
    #: call site is unaffected.
    is_context: bool = False


def ripgrep_available() -> bool:
    return shutil.which("rg") is not None


def _ignore_spec(root: str) -> pathspec.PathSpec[PathspecPattern]:
    lines = ["/.git/"]
    gitignore = Path(root) / ".gitignore"
    if gitignore.exists():
        lines.extend(gitignore.read_text(errors="replace").splitlines())
    return pathspec.PathSpec.from_lines(GLOB_PATTERN_FACTORY, lines)


def walk_files(
    root: str,
    *,
    pattern: str | None = None,
    limit: int = DEFAULT_LIMIT,
    keep: Callable[[str], bool] | None = None,
) -> list[str]:
    """Absolute paths under ``root``, gitignore-aware, most recently modified first.

    ``keep`` is asked about each path that would otherwise be listed, and a path it
    refuses is not. It is asked before ``limit`` is applied, not after: a refusal made
    on the capped list would let a directory of refused entries crowd out every other.
    A link to a directory is not entered, whatever it leads to.
    """
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
            if keep is not None and not keep(absolute):
                continue
            try:
                found.append((os.path.getmtime(absolute), absolute))  # noqa: PTH204
            except OSError:
                continue
    found.sort(reverse=True)
    return [path for _, path in found[:limit]]


def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse overlapping or touching (start, end) ranges into the fewest
    disjoint ones, each separated from its neighbor by at least one line.
    "Touching" (next start == this end + 1) merges too: two context windows
    with nothing but matched lines between them are one group, not two.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _with_context(
    path: str, lines: list[str], match_line_nos: list[int], before: int, after: int
) -> list[Match]:
    """Every line in the merged (before, after) windows around ``match_line_nos``,
    clipped to the file's own bounds, each tagged match or context. A line that
    both matches and falls in a neighboring match's window stays tagged as a
    match -- the ``not in match_set`` check, not the window it happened to be
    reached through, decides that.
    """
    total = len(lines)
    windows = [(max(1, n - before), min(total, n + after)) for n in match_line_nos]
    match_set = set(match_line_nos)
    result: list[Match] = []
    for start, end in _merge_windows(windows):
        for line_no in range(start, end + 1):
            result.append(
                Match(path, line_no, lines[line_no - 1], is_context=line_no not in match_set)
            )
    return result


def _apply_match_limit(entries: list[Match], limit: int, before: int, after: int) -> list[Match]:
    """head_limit counts matches, not lines (including context lines): this
    stops once ``limit`` matches have been kept, but it never cuts a group
    (a run of contiguous line numbers in the same file -- a match's own
    context window, plus any other match an overlapping window merged into
    the same group) off partway through just to land on the count exactly.
    The group containing the limit-th match, including every match inside
    that same group, is always emitted in full; only the next group is
    dropped. With no context requested (``before`` and ``after`` both 0),
    every entry is its own singleton group, so this is an exact cutoff at
    ``limit`` -- the original, pre-context behavior.
    """
    merging = bool(before or after)
    kept: list[Match] = []
    matches_seen = 0
    previous: tuple[str, int] | None = None
    for entry in entries:
        contiguous = (
            merging
            and previous is not None
            and previous[0] == entry.path
            and entry.line_no == previous[1] + 1
        )
        if not contiguous and matches_seen >= limit:
            break
        kept.append(entry)
        if not entry.is_context:
            matches_seen += 1
        previous = (entry.path, entry.line_no)
    return kept


def ripgrep_search(
    pattern: str,
    *,
    root: str,
    glob: str | None,
    ignore_case: bool = False,
    limit: int = DEFAULT_LIMIT,
    before: int = 0,
    after: int = 0,
    keep: Callable[[str], bool] | None = None,
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
        "--max-filesize",
        str(MAX_SEARCH_BYTES),
        "--glob",
        "!/.git/",
    ]
    if ignore_case:
        argv.append("-i")
    if before:
        argv += ["-B", str(before)]
    if after:
        argv += ["-A", str(after)]
    if glob:
        argv += ["--glob", glob]
    argv += ["--", pattern, root]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    if completed.returncode not in (0, 1):
        raise RuntimeError(completed.stderr.strip() or "ripgrep failed")
    matches: list[Match] = []
    for line in completed.stdout.splitlines():
        event = json.loads(line)
        event_type = event.get("type")
        if event_type not in ("match", "context"):
            continue
        data = event["data"]
        path = data["path"]["text"]
        # A path the caller does not want is dropped here, before the limit is applied
        # and not after: a refused file's matches must not use up a limit that the files
        # that are shown would have filled.
        if keep is not None and not keep(path):
            continue
        matches.append(
            Match(
                path,
                data["line_number"],
                data["lines"]["text"].rstrip("\n"),
                is_context=event_type == "context",
            )
        )
    # Not capped during the loop above: subprocess.run already waited for rg to
    # exit and buffered all of its output before this function saw any of it,
    # so breaking early here would save parsing a little JSON, not any
    # subprocess time -- and capping mid-stream is exactly how a match's own
    # trailing context gets cut off (see _apply_match_limit's docstring).
    return _apply_match_limit(matches, limit, before, after)


def python_search(
    pattern: str,
    *,
    root: str,
    glob: str | None,
    ignore_case: bool = False,
    limit: int = DEFAULT_LIMIT,
    before: int = 0,
    after: int = 0,
    keep: Callable[[str], bool] | None = None,
) -> list[Match]:
    regex = re.compile(pattern, re.IGNORECASE if ignore_case else 0)
    matches: list[Match] = []
    # keep is asked while walking, before a file is opened and before any limit applies: a
    # file the caller does not want is not read, and its matches cannot use up the limit.
    for path in sorted(walk_files(root, pattern=glob and f"**/{glob}", limit=10_000, keep=keep)):
        try:
            # lstat, not stat: a link is not a regular file, so one is not opened, which
            # is what ripgrep does and what keeps a link from reading a file it leads to.
            # Only a regular file is opened: reading a FIFO waits for a writer, and a
            # device may never end.
            status = Path(path).lstat()
            if not stat.S_ISREG(status.st_mode) or status.st_size > MAX_SEARCH_BYTES:
                continue
            data = Path(path).read_bytes()
        except OSError:
            continue
        if b"\0" in data[:BINARY_SNIFF_BYTES]:
            continue
        lines = data.decode("utf-8", errors="replace").splitlines()
        if not (before or after):
            # The original, pre-context fast path, left exactly as it was:
            # no windowing, no merging, an exact cutoff at limit. Equivalent
            # to the general path below (every match is its own singleton
            # group there too), but without building one-line windows and a
            # one-entry match_set per match for no reason.
            for line_no, line in enumerate(lines, 1):
                if regex.search(line):
                    matches.append(Match(path, line_no, line))
                    if len(matches) >= limit:
                        return matches
            continue
        file_matches = [n for n, line in enumerate(lines, 1) if regex.search(line)]
        if file_matches:
            matches.extend(_with_context(path, lines, file_matches, before, after))
            # A real match count, not len(matches) (which also counts context
            # lines): stopping once comfortably past the limit still lets
            # _apply_match_limit below find and complete the limit-th group,
            # while skipping files that could not matter -- a group never
            # spans more than one file (walk_files, and the gap check in
            # _apply_match_limit, are both per relative-to-root path).
            if sum(1 for m in matches if not m.is_context) >= limit:
                break
    if before or after:
        return _apply_match_limit(matches, limit, before, after)
    return matches


def search(
    pattern: str,
    *,
    root: str,
    glob: str | None = None,
    ignore_case: bool = False,
    limit: int = DEFAULT_LIMIT,
    before: int = 0,
    after: int = 0,
    keep: Callable[[str], bool] | None = None,
) -> list[Match]:
    """Matches of ``pattern`` under ``root``, with at most ``limit`` of them.

    ``keep`` is asked about the path of each file that has a match, and a file it refuses
    contributes nothing, matches or context, to the result or to the count the limit is
    applied to.
    """
    backend = ripgrep_search if ripgrep_available() else python_search
    return backend(
        pattern,
        root=root,
        glob=glob,
        ignore_case=ignore_case,
        limit=limit,
        before=before,
        after=after,
        keep=keep,
    )
