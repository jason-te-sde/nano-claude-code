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
