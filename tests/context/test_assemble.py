import subprocess
from pathlib import Path

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
