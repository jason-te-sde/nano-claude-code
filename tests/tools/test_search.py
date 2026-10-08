import os
import shutil
import time
from pathlib import Path

import pytest

from nanoclaude.tools import search as search_module
from nanoclaude.tools.search import (
    Match,
    python_search,
    ripgrep_available,
    ripgrep_search,
    walk_files,
)
from tests.fifo import call_without_blocking, make_fifo


@pytest.fixture
def corpus(tmp_repo):
    (tmp_repo / "src").mkdir()
    (tmp_repo / "src" / "a.py").write_text("def hello():\n    return 1\n")
    (tmp_repo / "src" / "b.py").write_text("def goodbye():\n    return hello()\n")
    (tmp_repo / "node_modules").mkdir()
    (tmp_repo / "node_modules" / "junk.py").write_text("def hello(): pass\n")
    (tmp_repo / "notes.md").write_text("hello world\n")
    return tmp_repo


def test_gitignored_directories_are_skipped(corpus):
    found = walk_files(str(corpus), pattern="**/*.py", limit=100)
    assert any("src/a.py" in f for f in found)
    assert not any("node_modules" in f for f in found)


def test_files_come_back_newest_first(corpus):
    os.utime(corpus / "src" / "b.py", (time.time() + 10, time.time() + 10))
    found = walk_files(str(corpus), pattern="**/*.py", limit=100)
    assert found[0].endswith("b.py")


def test_the_limit_is_respected(corpus):
    assert len(walk_files(str(corpus), pattern="**/*", limit=2)) == 2


def test_a_gitignored_file_pattern_is_skipped_outside_any_ignored_directory(corpus):
    """node_modules/ is pruned at the directory level before any file inside it
    is ever listed, so that alone never reaches the per-file ``ignore.match_file``
    check. *.pyc, a file-level pattern in tmp_repo's .gitignore, is the only way
    to reach it: a top-level file matched by name rather than by a pruned parent.
    """
    (corpus / "cache.pyc").write_text("binary-ish\n")
    found = walk_files(str(corpus), limit=100)
    assert not any(f.endswith("cache.pyc") for f in found)


def test_walk_files_works_without_a_gitignore(tmp_path):
    """_ignore_spec's ``if gitignore.exists()`` guard has two halves. ``tmp_repo``
    (used by every other test in this file) always creates a .gitignore, so this
    is the only place the "no .gitignore present" half is exercised.
    """
    (tmp_path / "a.py").write_text("x = 1\n")
    found = walk_files(str(tmp_path), pattern="**/*.py", limit=100)
    assert any(f.endswith("a.py") for f in found)


def test_a_file_whose_mtime_cannot_be_read_is_skipped(tmp_repo, monkeypatch):
    (tmp_repo / "flaky.py").write_text("x\n")
    real_getmtime = os.path.getmtime

    def flaky_getmtime(path: str) -> float:
        if str(path).endswith("flaky.py"):
            raise OSError("boom")
        return real_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", flaky_getmtime)
    found = walk_files(str(tmp_repo), limit=100)
    assert not any("flaky.py" in f for f in found)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            ripgrep_search,
            marks=pytest.mark.skipif(not ripgrep_available(), reason="ripgrep not on PATH"),
        ),
        python_search,
    ],
)
def test_both_backends_agree_on_a_simple_search(backend, corpus):
    """The fallback is not allowed to be a different tool with the same name."""
    matches = backend("def hello", root=str(corpus), glob="*.py", limit=100)
    assert [m.line_no for m in matches] == [1]
    assert matches[0].path.endswith("src/a.py")


@pytest.mark.parametrize("backend", [python_search])
def test_the_fallback_skips_binary_and_ignored_files(backend, corpus):
    (corpus / "blob.bin").write_bytes(b"\x00hello\x00")
    matches = backend("hello", root=str(corpus), glob=None, limit=100)
    assert not any("blob.bin" in m.path for m in matches)
    assert not any("node_modules" in m.path for m in matches)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            ripgrep_search,
            marks=pytest.mark.skipif(not ripgrep_available(), reason="ripgrep not on PATH"),
        ),
        python_search,
    ],
)
def test_the_match_limit_is_respected(backend, tmp_repo):
    (tmp_repo / "many.py").write_text("hello\n" * 10)
    matches = backend("hello", root=str(tmp_repo), glob="*.py", limit=3)
    assert len(matches) == 3


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            ripgrep_search,
            marks=pytest.mark.skipif(not ripgrep_available(), reason="ripgrep not on PATH"),
        ),
        python_search,
    ],
)
def test_ignore_case_matches_regardless_of_letter_case(backend, tmp_repo):
    (tmp_repo / "a.py").write_text("HELLO world\n")
    matches = backend("hello", root=str(tmp_repo), glob="*.py", ignore_case=True, limit=10)
    assert len(matches) == 1
    no_match = backend("hello", root=str(tmp_repo), glob="*.py", ignore_case=False, limit=10)
    assert no_match == []


def test_a_file_that_cannot_be_read_is_skipped_not_raised(corpus, monkeypatch):
    real_read_bytes = Path.read_bytes

    def flaky_read_bytes(self: Path, *args: object, **kwargs: object) -> bytes:
        if self.name == "a.py":
            raise OSError("boom")
        return real_read_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", flaky_read_bytes)
    matches = python_search("def hello", root=str(corpus), glob="*.py", limit=100)
    assert not any(m.path.endswith("a.py") for m in matches)


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not on PATH")
def test_ripgrep_search_raises_on_a_pattern_it_cannot_parse(corpus):
    """Python's ``re`` accepts backreferences; ripgrep's default engine does not
    (it suggests --pcre2 instead), so this is a real, naturally occurring
    divergence rather than a contrived failure -- and it is the only way
    ripgrep_search's own ``returncode not in (0, 1)`` branch fires.
    """
    with pytest.raises(RuntimeError, match="regex parse error"):
        ripgrep_search(r"(a)\1", root=str(corpus), glob=None, limit=10)


def test_ripgrep_available_reflects_whether_rg_is_on_path(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/rg")
    assert ripgrep_available() is True
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert ripgrep_available() is False


def test_search_uses_ripgrep_when_available(monkeypatch, corpus):
    used: list[str] = []

    def fake_ripgrep(*_args: object, **_kwargs: object) -> list[Match]:
        used.append("rg")
        return []

    def fake_python(*_args: object, **_kwargs: object) -> list[Match]:
        used.append("py")
        return []

    monkeypatch.setattr(search_module, "ripgrep_available", lambda: True)
    monkeypatch.setattr(search_module, "ripgrep_search", fake_ripgrep)
    monkeypatch.setattr(search_module, "python_search", fake_python)
    search_module.search("x", root=str(corpus), limit=10)
    assert used == ["rg"]


def test_search_falls_back_to_python_when_ripgrep_is_unavailable(monkeypatch, corpus):
    used: list[str] = []

    def fake_ripgrep(*_args: object, **_kwargs: object) -> list[Match]:
        used.append("rg")
        return []

    def fake_python(*_args: object, **_kwargs: object) -> list[Match]:
        used.append("py")
        return []

    monkeypatch.setattr(search_module, "ripgrep_available", lambda: False)
    monkeypatch.setattr(search_module, "ripgrep_search", fake_ripgrep)
    monkeypatch.setattr(search_module, "python_search", fake_python)
    search_module.search("x", root=str(corpus), limit=10)
    assert used == ["py"]


def test_match_is_context_defaults_to_false_so_old_call_sites_are_unchanged():
    assert Match("p", 1, "line").is_context is False


#: Context-aware behavior must agree between both backends. "ripgrep" is
#: skipped where rg is not on PATH; "python-fallback" is the other half.
CONTEXT_BACKENDS = [
    pytest.param(
        ripgrep_search,
        marks=pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not on PATH"),
        id="ripgrep",
    ),
    pytest.param(python_search, id="python-fallback"),
]


@pytest.mark.parametrize("backend", CONTEXT_BACKENDS)
def test_context_before_only(backend, tmp_repo):
    (tmp_repo / "a.py").write_text("line1\nline2\nMATCH\nline4\nline5\n")
    matches = backend("MATCH", root=str(tmp_repo), glob="*.py", limit=10, before=2, after=0)
    assert [(m.line_no, m.is_context) for m in matches] == [(1, True), (2, True), (3, False)]


@pytest.mark.parametrize("backend", CONTEXT_BACKENDS)
def test_context_after_only(backend, tmp_repo):
    (tmp_repo / "a.py").write_text("line1\nline2\nMATCH\nline4\nline5\n")
    matches = backend("MATCH", root=str(tmp_repo), glob="*.py", limit=10, before=0, after=2)
    assert [(m.line_no, m.is_context) for m in matches] == [(3, False), (4, True), (5, True)]


@pytest.mark.parametrize("backend", CONTEXT_BACKENDS)
def test_context_both_sides(backend, tmp_repo):
    (tmp_repo / "a.py").write_text("line1\nline2\nMATCH\nline4\nline5\n")
    matches = backend("MATCH", root=str(tmp_repo), glob="*.py", limit=10, before=1, after=1)
    assert [(m.line_no, m.is_context) for m in matches] == [(2, True), (3, False), (4, True)]


@pytest.mark.parametrize("backend", CONTEXT_BACKENDS)
def test_overlapping_windows_merge_without_duplicate_lines(backend, tmp_repo):
    """MATCH_A (line 2) and MATCH_B (line 4), before=after=2: windows (1,4) and
    (2,5) overlap and must merge into one (1,5) run with each line exactly once,
    and the line that is itself MATCH_B must stay tagged as a match, not be
    demoted to context merely because it also falls inside MATCH_A's window.
    """
    (tmp_repo / "a.py").write_text("l1\nMATCH_A\nl3\nMATCH_B\nl5\n")
    matches = backend("MATCH", root=str(tmp_repo), glob="*.py", limit=10, before=2, after=2)
    assert [(m.line_no, m.is_context) for m in matches] == [
        (1, True),
        (2, False),
        (3, True),
        (4, False),
        (5, True),
    ]
    assert len({m.line_no for m in matches}) == len(matches), "a line was printed twice"


@pytest.mark.parametrize("backend", CONTEXT_BACKENDS)
def test_windows_are_clipped_at_the_start_of_a_file(backend, tmp_repo):
    (tmp_repo / "a.py").write_text("MATCH\nline2\n")
    matches = backend("MATCH", root=str(tmp_repo), glob="*.py", limit=10, before=5, after=0)
    assert [m.line_no for m in matches] == [1]


@pytest.mark.parametrize("backend", CONTEXT_BACKENDS)
def test_windows_are_clipped_at_the_end_of_a_file(backend, tmp_repo):
    (tmp_repo / "a.py").write_text("line1\nMATCH\n")
    matches = backend("MATCH", root=str(tmp_repo), glob="*.py", limit=10, before=0, after=5)
    assert [m.line_no for m in matches] == [2]


@pytest.mark.parametrize("backend", CONTEXT_BACKENDS)
def test_the_match_limit_keeps_a_whole_group_intact_when_context_is_requested(backend, tmp_repo):
    """head_limit counts matches, not lines (see grep.py's _apply_match_limit):
    once the limit-th match's group has started, that whole group -- including
    a second match merged into it by an overlapping window -- is still emitted
    in full, and only the *next* group is dropped. This is what lets a match's
    own context never be printed partway through.
    """
    (tmp_repo / "a.py").write_text("l1\nMATCH_A\nl3\nMATCH_B\nl5\nl6\nl7\nl8\nMATCH_C\nl10\n")
    matches = backend("MATCH", root=str(tmp_repo), glob="*.py", limit=1, before=1, after=1)
    assert [m.line_no for m in matches] == [1, 2, 3, 4, 5]
    assert sum(1 for m in matches if not m.is_context) == 2


@pytest.mark.parametrize("backend", CONTEXT_BACKENDS)
def test_the_match_limit_without_context_is_an_exact_cutoff(backend, tmp_repo):
    """The no-context case (before=after=0) must still cut off at exactly
    `limit` matches, even when matches sit on consecutive lines -- two adjacent
    matching lines must not be treated as one "group" when no context was ever
    requested, which is exactly what the regression below pins.
    """
    (tmp_repo / "a.py").write_text("hello\n" * 10)
    matches = backend("hello", root=str(tmp_repo), glob="*.py", limit=3, before=0, after=0)
    assert len(matches) == 3


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not on PATH")
def test_both_backends_produce_identical_context_output(tmp_repo):
    (tmp_repo / "a.py").write_text("l1\nMATCH_A\nl3\nMATCH_B\nl5\nl6\nl7\nl8\nMATCH_C\nl10\n")
    rg_matches = ripgrep_search(
        "MATCH", root=str(tmp_repo), glob="*.py", limit=10, before=1, after=1
    )
    py_matches = python_search(
        "MATCH", root=str(tmp_repo), glob="*.py", limit=10, before=1, after=1
    )

    def as_tuples(matches: list[Match]) -> list[tuple[int, str, bool]]:
        return [(m.line_no, m.line, m.is_context) for m in matches]

    assert as_tuples(rg_matches) == as_tuples(py_matches)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            ripgrep_search,
            marks=pytest.mark.skipif(not ripgrep_available(), reason="ripgrep not on PATH"),
        ),
        python_search,
    ],
)
def test_both_backends_skip_files_over_the_size_cap(backend, corpus, monkeypatch):
    monkeypatch.setattr("nanoclaude.tools.search.MAX_SEARCH_BYTES", 64)
    (corpus / "big.txt").write_text("needle\n" + "x" * 100 + "\n")
    (corpus / "small.txt").write_text("needle\n")
    found = {Path(m.path).name for m in backend("needle", root=str(corpus), glob=None, limit=100)}
    assert "small.txt" in found
    assert "big.txt" not in found


def test_the_fallback_search_skips_a_fifo_rather_than_waiting_on_it(tmp_path):
    # Reading a FIFO waits for a writer, so a search that opened every path it walked
    # would stop at the first one in the tree. Only regular files are searched.
    (tmp_path / "a.py").write_text("def hello():\n    pass\n")
    fifo = make_fifo(tmp_path / "b.py")
    matches = call_without_blocking(
        fifo, python_search, "def hello", root=str(tmp_path), glob="*.py", limit=100
    )
    assert [Path(m.path).name for m in matches] == ["a.py"]


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            ripgrep_search,
            marks=pytest.mark.skipif(not ripgrep_available(), reason="ripgrep not on PATH"),
        ),
        python_search,
    ],
)
def test_neither_backend_reads_through_a_symlink(backend, tmp_path):
    """ripgrep does not follow links, so a file reached only through one is not
    searched. The fallback used to stat and read through them, which is how a link
    inside the project to a file outside it had its lines returned. What a link
    leads to is searched under its own name when it is in the project at all, so
    nothing is lost by not following it.
    """
    project, outside = tmp_path / "project", tmp_path / "outside"
    (project / "real").mkdir(parents=True)
    outside.mkdir()
    (outside / "private.txt").write_text("needle outside\n")
    (project / "real" / "inside.txt").write_text("needle inside\n")
    (project / "file_link.txt").symlink_to(outside / "private.txt")
    (project / "dir_link").symlink_to(outside, target_is_directory=True)
    (project / "inside_link.txt").symlink_to(project / "real" / "inside.txt")
    (project / "inside_dir_link").symlink_to(project / "real", target_is_directory=True)

    found = [m.path for m in backend("needle", root=str(project), glob=None, limit=100)]

    assert [Path(p).relative_to(project).as_posix() for p in found] == ["real/inside.txt"]
