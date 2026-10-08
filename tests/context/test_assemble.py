import ast
import os
import re
import subprocess
from pathlib import Path

import pytest

from nanoclaude.context.assemble import assemble, environment_block, git_state


def test_the_system_prompt_is_identical_across_two_assemblies(tmp_repo):
    """Prompt caching needs a byte-stable prefix; a timestamp here would cost it."""
    first = assemble(str(tmp_repo), cwd=str(tmp_repo), home=None)
    second = assemble(str(tmp_repo), cwd=str(tmp_repo), home=None)
    assert first.system == second.system


def test_git_state_is_in_the_environment_block_not_the_system_prompt(tmp_repo):
    """It changes during a session, so it must sit outside the cached prefix."""
    context = assemble(str(tmp_repo), cwd=str(tmp_repo), home=None)
    assert "git branch" not in context.system


def test_instructions_reach_the_system_prompt(tmp_repo):
    (tmp_repo / "NANO.md").write_text("Never touch migrations/.\n")
    assert "migrations" in assemble(str(tmp_repo), cwd=str(tmp_repo), home=None).system


def test_a_directory_that_is_not_a_git_repository_still_works(tmp_path):
    assert isinstance(environment_block(str(tmp_path)), str)


def _init_real_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=path, check=True)  # noqa: S607
    # Built as a variable rather than inlined: unlike a literal list, this is
    # not a static "partial executable path" ruff can flag (S607), the same
    # reasoning tools/search.py's own argv list already relies on.
    commit = [
        "git",
        "-c",
        "user.email=t@example.com",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-q",
        "-m",
        "init",
    ]
    subprocess.run(commit, cwd=path, check=True)  # noqa: S603


def test_git_state_reports_the_branch_and_the_dirty_count(tmp_repo):
    """tmp_repo's own ".git" (conftest.py) is an empty placeholder directory,
    which git refuses to recognise as a repository at all (confirmed by hand:
    "fatal: not a git repository"). So nothing above actually runs
    git_state's non-empty-branch line -- this builds a real repository so
    that it does.
    """
    _init_real_repo(tmp_repo)
    state = git_state(str(tmp_repo))
    assert state == "git branch: main (1 file(s) with uncommitted changes)"


def test_real_git_state_stays_out_of_the_system_prompt_and_in_the_environment(tmp_repo):
    """The non-vacuous version of the "not the system prompt" test above: its
    git_state is always empty there (see that test's own docstring), so its
    assertion holds for a reason that has nothing to do with the split it
    claims to verify. This repeats it with real content.
    """
    _init_real_repo(tmp_repo)
    context = assemble(str(tmp_repo), cwd=str(tmp_repo), home=None)
    assert "git branch" not in context.system
    assert "git branch: main" in context.environment


def test_a_missing_git_executable_produces_no_traceback(tmp_repo, monkeypatch, tmp_path_factory):
    """Make sure of: environment_block must not raise when git is not installed
    at all, not just when it is installed but finds no repository (the
    not-a-git-repository test above) -- subprocess.run raises FileNotFoundError
    in that case, which git_state must also catch.
    """
    empty_bin = tmp_path_factory.mktemp("emptybin")
    monkeypatch.setenv("PATH", str(empty_bin))
    block = environment_block(str(tmp_repo))
    assert "git branch" not in block


# ---- a repository's own configuration must not make assembling context run a program

GIT_USER = ["-c", "user.email=t@example.com", "-c", "user.name=t"]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *GIT_USER, *args], cwd=repo, check=True, capture_output=True)  # noqa: S603, S607


def _script(path: Path, marker: Path, *, then: str = "") -> Path:
    """An executable that leaves ``marker`` behind when something runs it."""
    path.write_text(f"#!/bin/sh\necho ran >> '{marker}'\n{then}\n")
    path.chmod(0o755)
    return path


def _git_at_least(major: int, minor: int) -> bool:
    out = subprocess.run(["git", "--version"], capture_output=True, text=True, check=True).stdout  # noqa: S607
    found = re.search(r"(\d+)\.(\d+)", out)
    assert found, out
    return (int(found[1]), int(found[2])) >= (major, minor)


def _make_stale(path: Path) -> None:
    """Change the file's stat data and not its content, as a touch does: git then has to
    look inside it to know whether it changed, and to refresh what the index remembers."""
    moved = path.stat().st_mtime_ns + 10_000_000_000
    os.utime(path, ns=(moved, moved))


def _assert_runs_it_plain_and_not_when_assembling(
    repo: Path, marker: Path, *, stale: Path | None = None
) -> None:
    """The control first: plain git runs the program, so a marker that is absent
    afterwards means the program was kept from running and not that git never would.
    The control refreshes the index, so the file is made stale again before assembling."""
    if stale is not None:
        _make_stale(stale)
    subprocess.run(["git", "status", "--porcelain"], cwd=repo, check=True, capture_output=True)  # noqa: S607
    assert marker.exists(), "plain git status did not run the program; the test proves nothing"
    marker.unlink()

    if stale is not None:
        _make_stale(stale)
    context = assemble(str(repo), cwd=str(repo), home=None)

    assert not marker.exists(), "assembling the context ran a program the repository named"
    # Not kept from running by git failing altogether.
    assert "git branch: main" in context.environment


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    (path / "a.txt").write_text("hello\n")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "init")
    return path


def test_assembling_context_does_not_run_the_fsmonitor_a_repository_names(repo, tmp_path):
    marker = tmp_path / "marker"
    monitor = _script(tmp_path / "monitor.sh", marker)
    _git(repo, "config", "core.fsmonitor", str(monitor))
    _assert_runs_it_plain_and_not_when_assembling(repo, marker)


def test_assembling_context_does_not_run_a_hook_the_index_refresh_would_fire(repo, tmp_path):
    """``git status`` opportunistically rewrites a stale index, and writing the index fires
    ``post-index-change``. With no lock taken it does not write, so nothing fires."""
    marker = tmp_path / "marker"
    _script(repo / ".git" / "hooks" / "post-index-change", marker)
    _assert_runs_it_plain_and_not_when_assembling(repo, marker, stale=repo / "a.txt")


@pytest.mark.skipif(
    not _git_at_least(2, 40),
    reason="git before 2.40 has no GIT_ATTR_SOURCE to switch attributes off",
)
def test_assembling_context_does_not_run_a_clean_filter_a_repository_names(repo, tmp_path):
    """``status`` compares a file whose stat data changed by running its clean filter, and
    a filter is a program named by the repository's config and chosen by its attributes."""
    marker = tmp_path / "marker"
    cleaner = _script(tmp_path / "clean.sh", marker, then="cat")
    (repo / ".gitattributes").write_text("a.txt filter=evil\n")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-q", "-m", "attributes")
    _git(repo, "config", "filter.evil.clean", str(cleaner))
    _assert_runs_it_plain_and_not_when_assembling(repo, marker, stale=repo / "a.txt")


def test_the_dirty_count_is_still_right_with_the_overrides_in_force(repo):
    (repo / "a.txt").write_text("changed\n")
    (repo / "new.txt").write_text("new\n")
    assert git_state(str(repo)) == "git branch: main (2 file(s) with uncommitted changes)"


def test_git_state_is_empty_rather_than_wrong_when_status_fails(repo, monkeypatch):
    """No count is better than a count of zero that only means the question failed."""
    import nanoclaude.context.git as gitmod

    real = gitmod.run_git

    def failing_status(root: str, *args: str) -> subprocess.CompletedProcess[str]:
        if args and args[0] == "status":
            return subprocess.CompletedProcess(args, 128, stdout="", stderr="fatal: boom")
        return real(root, *args)

    monkeypatch.setattr("nanoclaude.context.assemble.run_git", failing_status)
    assert git_state(str(repo)) == ""


def test_git_is_run_from_one_place():
    """Every git invocation ncc makes goes through context/git.py's run_git, which holds
    the overrides. A second place that spells out ``"git"`` as a command is a way around
    them."""
    src = Path(__file__).resolve().parents[2] / "src" / "nanoclaude"
    home = src / "context" / "git.py"
    assert home.is_file()
    offenders = [
        str(path.relative_to(src))
        for path in sorted(src.rglob("*.py"))
        if path != home
        and any(
            isinstance(node, ast.Constant) and node.value == "git"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
    ]
    assert offenders == []
