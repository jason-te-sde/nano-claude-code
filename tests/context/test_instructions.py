import subprocess
from pathlib import Path

import pytest

from nanoclaude.context.instructions import (
    INSTRUCTION_FILENAMES,
    MAX_INSTRUCTION_BYTES,
    load_instructions,
)


def test_instruction_filenames_puts_nano_before_claude():
    """The order is the priority: _read_one returns on the first match it finds,
    so if this tuple were ever reordered, CLAUDE.md would silently start
    winning over NANO.md whenever both exist.
    """
    assert INSTRUCTION_FILENAMES == ("NANO.md", "CLAUDE.md")


def test_nano_md_at_the_root_is_loaded(tmp_repo):
    (tmp_repo / "NANO.md").write_text("Always run make lint.\n")
    assert "make lint" in load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=None)


def test_claude_md_is_read_when_nano_md_is_absent(tmp_repo):
    (tmp_repo / "CLAUDE.md").write_text("Use tabs.\n")
    assert "Use tabs" in load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=None)


def test_nano_md_wins_when_both_exist(tmp_repo):
    (tmp_repo / "NANO.md").write_text("nano wins\n")
    (tmp_repo / "CLAUDE.md").write_text("claude loses\n")
    loaded = load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=None)
    assert "nano wins" in loaded and "claude loses" not in loaded


def test_a_nested_directory_layers_on_top_of_the_root(tmp_repo):
    (tmp_repo / "NANO.md").write_text("root rule\n")
    nested = tmp_repo / "service"
    nested.mkdir()
    (nested / "NANO.md").write_text("service rule\n")
    loaded = load_instructions(str(tmp_repo), cwd=str(nested), home=None)
    assert loaded.index("root rule") < loaded.index("service rule")


def test_an_enormous_instructions_file_is_truncated_not_loaded_whole(tmp_repo):
    """Review Focus #4. A 400 KB NANO.md would consume the whole window."""
    (tmp_repo / "NANO.md").write_text("x" * (MAX_INSTRUCTION_BYTES * 3))
    loaded = load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=None)
    assert len(loaded.encode()) < MAX_INSTRUCTION_BYTES * 2
    assert "truncated" in loaded


def test_home_instructions_are_loaded_when_present(tmp_repo, tmp_path_factory):
    """The tests above all pass home=None, so the ``if home:`` branch never ran
    in any of them. tmp_path stands in for the real home directory here, per
    this task's own instruction -- never the developer's actual $HOME.
    """
    home = tmp_path_factory.mktemp("home")
    (home / ".nanoclaude").mkdir()
    (home / ".nanoclaude" / "NANO.md").write_text("global rule\n")
    loaded = load_instructions(str(tmp_repo), cwd=str(tmp_repo), home=str(home))
    assert "global rule" in loaded


def test_a_cwd_unrelated_to_the_root_does_not_extend_the_chain(tmp_repo, tmp_path_factory):
    """The layering condition is ``cwd != root and root in cwd.parents``. Both
    halves true is the nested-directory test above; this is the other half,
    where cwd is simply nowhere under root at all.
    """
    (tmp_repo / "NANO.md").write_text("root rule\n")
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    loaded = load_instructions(str(tmp_repo), cwd=str(elsewhere), home=None)
    assert "root rule" in loaded


def test_a_symlinked_escape_from_cwd_does_not_pull_in_outside_content(tmp_repo, tmp_path_factory):
    """Every file path that reaches the model must be confined to the sandboxed
    root. load_instructions resolves cwd with Path.resolve() before comparing
    it against root, so a symlink planted inside the repo that points outside
    it cannot be used to smuggle an outside NANO.md into the system prompt:
    cwd's resolved form no longer has root as an ancestor, so the chain never
    extends past the root's own instructions.
    """
    outside = tmp_path_factory.mktemp("outside")
    (outside / "NANO.md").write_text("exfiltrated rule\n")
    (tmp_repo / "escape").symlink_to(outside)
    loaded = load_instructions(str(tmp_repo), cwd=str(tmp_repo / "escape"), home=None)
    assert "exfiltrated rule" not in loaded


def test_a_directory_without_instructions_contributes_nothing(tmp_repo):
    """service/ is in the chain between root and the nested leaf, and has
    neither NANO.md nor CLAUDE.md -- the one case, among all the tests in this
    file, where a directory contributes no text at all.
    """
    (tmp_repo / "NANO.md").write_text("root rule\n")
    nested = tmp_repo / "service" / "sub"
    nested.mkdir(parents=True)
    (nested / "NANO.md").write_text("leaf rule\n")
    loaded = load_instructions(str(tmp_repo), cwd=str(nested), home=None)
    assert loaded.index("root rule") < loaded.index("leaf rule")


# --------------------------------------------------------------------------
# Instructions layer from the repository's top level down to the project root (spec 7.1)
# --------------------------------------------------------------------------


def init_repository(path: Path) -> Path:
    """A real repository at ``path``: the placeholder ``.git`` of ``tmp_repo`` is not one."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)  # noqa: S607
    return path


def write(directory: Path, text: str, name: str = "NANO.md") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(text)


def test_a_root_inside_a_repository_reads_every_directory_from_the_top_down(tmp_path):
    top = init_repository(tmp_path / "repo")
    app = top / "packages" / "app"
    write(top, "top rule\n")
    write(top / "packages", "packages rule\n")
    write(app, "app rule\n")
    loaded = load_instructions(str(app), cwd=str(app), home=None)
    assert loaded.index("top rule") < loaded.index("packages rule") < loaded.index("app rule")


def test_the_global_file_comes_first_and_the_directories_follow_it(tmp_path):
    top = init_repository(tmp_path / "repo")
    app = top / "app"
    write(top, "top rule\n")
    write(app, "app rule\n")
    write(tmp_path / "home" / ".nanoclaude", "global rule\n")
    loaded = load_instructions(str(app), cwd=str(app), home=str(tmp_path / "home"))
    assert loaded.index("global rule") < loaded.index("top rule") < loaded.index("app rule")


def test_the_chain_stops_at_the_top_of_the_repository(tmp_path):
    write(tmp_path, "above the repository\n")
    top = init_repository(tmp_path / "repo")
    app = top / "app"
    write(top, "top rule\n")
    write(app, "app rule\n")
    loaded = load_instructions(str(app), cwd=str(app), home=None)
    assert "top rule" in loaded and "above the repository" not in loaded


def test_outside_a_repository_only_the_root_is_read(tmp_path):
    write(tmp_path, "the parent's rule\n")
    root = tmp_path / "plain"
    write(root, "root rule\n")
    loaded = load_instructions(str(root), cwd=str(root), home=None)
    assert "root rule" in loaded and "the parent's rule" not in loaded


def test_a_root_that_is_the_top_is_read_once(tmp_path):
    top = init_repository(tmp_path / "repo")
    write(top, "top rule\n")
    loaded = load_instructions(str(top), cwd=str(top), home=None)
    assert loaded.count("top rule") == 1


def test_a_directory_in_the_chain_with_nothing_in_it_contributes_nothing(tmp_path):
    top = init_repository(tmp_path / "repo")
    app = top / "packages" / "app"
    write(top, "top rule\n")
    write(app, "app rule\n")
    loaded = load_instructions(str(app), cwd=str(app), home=None)
    assert (
        loaded.count("<!--") == 2
    )  # a header each for the two files there are, none for packages/


def test_each_directory_in_the_chain_may_keep_its_instructions_under_either_name(tmp_path):
    top = init_repository(tmp_path / "repo")
    app = top / "app"
    write(top, "top is a CLAUDE.md\n", name="CLAUDE.md")
    write(app, "app is a NANO.md\n")
    write(app, "and its CLAUDE.md is ignored\n", name="CLAUDE.md")
    loaded = load_instructions(str(app), cwd=str(app), home=None)
    assert "top is a CLAUDE.md" in loaded and "app is a NANO.md" in loaded
    assert "its CLAUDE.md is ignored" not in loaded


def test_a_root_reached_through_a_symlink_is_placed_in_the_repository_it_leads_to(tmp_path):
    top = init_repository(tmp_path / "repo")
    app = top / "app"
    write(top, "top rule\n")
    write(app, "app rule\n")
    link = tmp_path / "link"
    link.symlink_to(app)
    loaded = load_instructions(str(link), cwd=str(link), home=None)
    assert loaded.index("top rule") < loaded.index("app rule")


def test_the_top_is_found_through_the_one_safe_way_of_running_git(tmp_path, monkeypatch):
    top = init_repository(tmp_path / "repo")
    app = top / "app"
    write(top, "top rule\n")
    write(app, "app rule\n")
    asked: list[tuple[str, ...]] = []

    def fail(root: str, *args: str, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        asked.append((root, *args))
        raise OSError("no git here")

    monkeypatch.setattr("nanoclaude.context.instructions.run_git", fail)
    loaded = load_instructions(str(app), cwd=str(app), home=None)
    assert asked == [(str(app.resolve()), "rev-parse", "--show-toplevel")]
    assert "app rule" in loaded and "top rule" not in loaded


@pytest.mark.parametrize(
    "failure",
    [OSError("git is not installed"), subprocess.TimeoutExpired("git", 5.0)],
    ids=["no git", "timeout"],
)
def test_when_git_cannot_answer_only_the_root_is_read_and_nothing_is_raised(
    tmp_path, monkeypatch, failure
):
    top = init_repository(tmp_path / "repo")
    app = top / "app"
    write(top, "top rule\n")
    write(app, "app rule\n")

    def cannot(*_args: object, **_kwargs: object) -> None:
        raise failure

    monkeypatch.setattr("nanoclaude.context.instructions.run_git", cannot)
    loaded = load_instructions(str(app), cwd=str(app), home=None)
    assert "app rule" in loaded and "top rule" not in loaded


def test_an_answer_from_a_git_that_failed_is_not_trusted_for_the_path_it_printed(
    tmp_path, monkeypatch
):
    top = init_repository(tmp_path / "repo")
    app = top / "app"
    write(top, "top rule\n")
    write(app, "app rule\n")

    def failed_but_printed(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 128, stdout=f"{top}\n", stderr="fatal: oops")

    monkeypatch.setattr("nanoclaude.context.instructions.run_git", failed_but_printed)
    loaded = load_instructions(str(app), cwd=str(app), home=None)
    assert "app rule" in loaded and "top rule" not in loaded
