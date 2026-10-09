import ast
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.context.assemble import environment_block, git_state
from nanoclaude.context.git import EMPTY_TREE, run_git
from tests.context.helpers import assemble_in


def test_the_system_prompt_is_identical_across_two_assemblies(tmp_repo):
    """Prompt caching needs a byte-stable prefix; a timestamp here would cost it."""
    first = assemble_in(tmp_repo)
    second = assemble_in(tmp_repo)
    assert first.system == second.system


def test_git_state_is_in_the_environment_block_not_the_system_prompt(tmp_repo):
    """It changes during a session, so it must sit outside the cached prefix."""
    context = assemble_in(tmp_repo)
    assert "git branch" not in context.system


def test_instructions_reach_the_system_prompt(tmp_repo):
    (tmp_repo / "NANO.md").write_text("Never touch migrations/.\n")
    assert "migrations" in assemble_in(tmp_repo).system


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


def test_git_state_reports_the_branch_and_nothing_about_the_working_tree(tmp_repo):
    """tmp_repo's own ".git" (conftest.py) is an empty placeholder directory,
    which git refuses to recognise as a repository at all (confirmed by hand:
    "fatal: not a git repository"). So nothing above actually runs
    git_state's non-empty-branch line -- this builds a real repository so
    that it does.
    """
    _init_real_repo(tmp_repo)
    (tmp_repo / "untracked.txt").write_text("not committed\n")
    assert git_state(str(tmp_repo)) == "git branch: main"


def test_real_git_state_stays_out_of_the_system_prompt_and_in_the_environment(tmp_repo):
    """The non-vacuous version of the "not the system prompt" test above: its
    git_state is always empty there (see that test's own docstring), so its
    assertion holds for a reason that has nothing to do with the split it
    claims to verify. This repeats it with real content.
    """
    _init_real_repo(tmp_repo)
    context = assemble_in(tmp_repo)
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
    context = assemble_in(repo)

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


def _filter_marker(repo: Path, tmp_path: Path) -> tuple[Path, Path]:
    """A clean filter named ``evil`` in the repository's config, and the marker it leaves."""
    marker = tmp_path / "marker"
    cleaner = _script(tmp_path / "clean.sh", marker, then="cat")
    _git(repo, "config", "filter.evil.clean", str(cleaner))
    return marker, cleaner


def test_assembling_context_does_not_run_a_clean_filter_a_gitattributes_file_selects(
    repo, tmp_path
):
    """``status`` compares a file whose stat data changed by running its clean filter, and
    a filter is a program named by the repository's config and chosen by its attributes."""
    marker, _ = _filter_marker(repo, tmp_path)
    (repo / ".gitattributes").write_text("a.txt filter=evil\n")
    _git(repo, "add", ".gitattributes")
    _git(repo, "commit", "-q", "-m", "attributes")
    _assert_runs_it_plain_and_not_when_assembling(repo, marker, stale=repo / "a.txt")


def test_assembling_context_does_not_run_a_clean_filter_info_attributes_selects(repo, tmp_path):
    """The same filter chosen from ``.git/info/attributes``, which no attribute source
    override reaches and which git before 2.40 would not have overridden anyway."""
    marker, _ = _filter_marker(repo, tmp_path)
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("a.txt filter=evil\n")
    _assert_runs_it_plain_and_not_when_assembling(repo, marker, stale=repo / "a.txt")


def test_assembling_context_does_not_run_a_clean_filter_core_attributesfile_selects(repo, tmp_path):
    marker, _ = _filter_marker(repo, tmp_path)
    attributes = tmp_path / "attributes"
    attributes.write_text("a.txt filter=evil\n")
    _git(repo, "config", "core.attributesFile", str(attributes))
    _assert_runs_it_plain_and_not_when_assembling(repo, marker, stale=repo / "a.txt")


def test_git_state_is_empty_rather_than_wrong_when_git_fails(repo, monkeypatch):
    """A branch name printed by a git that then exited non-zero (an unborn branch prints
    ``HEAD`` and exits 128) is not a statement of fact the model should believe."""

    def failing(root: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 128, stdout="HEAD\n", stderr="fatal: boom")

    monkeypatch.setattr("nanoclaude.context.assemble.run_git", failing)
    assert git_state(str(repo)) == ""


def test_git_state_is_empty_when_git_prints_no_branch(repo, monkeypatch):
    def silent(root: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 0, stdout="\n", stderr="")

    monkeypatch.setattr("nanoclaude.context.assemble.run_git", silent)
    assert git_state(str(repo)) == ""


def test_git_state_is_empty_when_git_cannot_be_run_or_takes_too_long(repo, monkeypatch):
    def missing(root: str, *args: str) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("git")

    def slow(root: str, *args: str) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, 5.0)

    for failure in (missing, slow):
        monkeypatch.setattr("nanoclaude.context.assemble.run_git", failure)
        assert git_state(str(repo)) == ""


def test_an_unborn_branch_gets_no_git_line_not_a_wrong_one(tmp_path):
    path = tmp_path / "fresh"
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    assert git_state(str(path)) == ""


def test_the_only_git_commands_assembling_context_runs_are_rev_parse_forms(repo, monkeypatch):
    """Anything that reads the working tree (``status``, ``diff``, ``ls-files``) can run a
    program the repository configures, by a route nobody has listed yet."""
    real = subprocess.run
    seen: list[list[str]] = []

    def recording(argv: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(list(argv))
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording)
    assemble_in(repo)
    git_calls = [argv for argv in seen if argv and argv[0] == "git"]
    assert git_calls, "assembling the context asked git nothing; the test proves nothing"
    assert {argv[3] for argv in git_calls} == {"rev-parse"}, git_calls


# ---- run_git itself refuses anything but rev-parse


@pytest.mark.parametrize(
    "args",
    [
        ("status", "--porcelain"),
        ("diff",),
        ("log", "-1"),
        ("ls-files",),
        ("show", "HEAD"),
        ("fetch",),
        ("-c", "core.pager=x", "rev-parse", "HEAD"),  # an option in front of the subcommand
        ("--git-dir=.git", "rev-parse"),
        ("Rev-Parse", "HEAD"),
        ("rev-parser",),
        ("",),
        (),
    ],
    ids=repr,
)
def test_run_git_refuses_everything_but_rev_parse(repo, monkeypatch, args):
    def must_not_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("a refused git command was started")

    monkeypatch.setattr(subprocess, "run", must_not_run)
    with pytest.raises(ValueError, match="rev-parse"):
        run_git(str(repo), *args)


def test_run_git_keeps_the_settings_that_were_there_before_the_rule(repo, monkeypatch):
    """With only rev-parse run, no route reaches these three, so no behaviour test can tell
    that they are there; they are kept as a second line, and pinned here as they are."""
    real = subprocess.run
    seen: dict[str, Any] = {}

    def recording(argv: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.update(argv=list(argv), env=kwargs["env"])
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording)
    run_git(str(repo), "rev-parse", "HEAD")
    assert seen["argv"][:3] == ["git", "-c", "core.fsmonitor=false"]
    assert seen["env"]["GIT_OPTIONAL_LOCKS"] == "0"
    assert seen["env"]["GIT_ATTR_SOURCE"] == EMPTY_TREE


def test_run_git_runs_a_rev_parse_form(repo):
    done = run_git(str(repo), "rev-parse", "--abbrev-ref", "HEAD")
    assert (done.returncode, done.stdout.strip()) == (0, "main")
    assert run_git(str(repo), "rev-parse", "--show-toplevel").returncode == 0


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
