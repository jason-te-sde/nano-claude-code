import os

import pytest

from nanoclaude.context.projectmap import project_map


def test_the_map_shows_the_tree_and_respects_gitignore(tmp_repo):
    (tmp_repo / "src").mkdir()
    (tmp_repo / "src" / "main.py").write_text("")
    (tmp_repo / "node_modules").mkdir()
    (tmp_repo / "node_modules" / "junk.js").write_text("")
    rendered = project_map(str(tmp_repo))
    assert "src/" in rendered and "main.py" in rendered
    assert "node_modules" not in rendered


def test_depth_is_limited(tmp_repo):
    deep = tmp_repo / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("")
    rendered = project_map(str(tmp_repo), depth=2)
    assert "deep.py" not in rendered


def test_a_symlink_loop_does_not_hang(tmp_repo):
    """Review Focus #4. Ordinary in real repositories, fatal if walked naively."""
    (tmp_repo / "sub").mkdir()
    (tmp_repo / "sub" / "loop").symlink_to(tmp_repo)
    rendered = project_map(str(tmp_repo), depth=5)
    assert isinstance(rendered, str)


def test_the_entry_count_is_capped_and_says_so(tmp_repo):
    for index in range(500):
        (tmp_repo / f"f{index}.py").write_text("")
    rendered = project_map(str(tmp_repo), max_entries=50)
    assert rendered.count("\n") <= 60
    assert "truncated" in rendered


def test_no_gitignore_file_means_only_git_itself_is_hidden(tmp_path):
    """tmp_repo (used by every test above) always creates a .gitignore, so none
    of them exercise the case where one is absent. Without a .gitignore, the
    map must still hide .git -- the one rule that is hardcoded, not read from
    the file -- and must not crash looking for a file that is not there.
    """
    (tmp_path / ".git").mkdir()
    (tmp_path / "a.py").write_text("")
    rendered = project_map(str(tmp_path))
    assert "a.py" in rendered and ".git" not in rendered


def test_a_root_that_does_not_exist_does_not_crash(tmp_path):
    """directory.stat() in walk()'s very first call -- on the root itself, with
    no prior is_dir() check from a parent iteration to have already ruled this
    out -- can fail just as surely as it can one level down. A non-existent
    root is the simplest deterministic way to reach that except branch.
    """
    rendered = project_map(str(tmp_path / "does-not-exist"))
    assert rendered == ""


@pytest.mark.skipif(os.geteuid() == 0, reason="root can list a mode-000 directory")
def test_an_unreadable_subdirectory_is_listed_but_not_descended_into(tmp_repo):
    """iterdir() can fail even when stat() on that same directory already
    succeeded: listing needs execute permission on the directory itself,
    stat'ing it from the parent's loop only needs search permission on the
    parent. A locked-down subdirectory reaches the second except branch
    without ever touching the first.
    """
    locked = tmp_repo / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        rendered = project_map(str(tmp_repo))
        assert "locked" in rendered  # listed, even though its contents could not be
    finally:
        locked.chmod(0o700)


def test_a_symlink_to_a_directory_outside_the_project_is_listed_but_not_entered(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "private-notes.txt").write_text("x")
    (project / "vendor").symlink_to(outside, target_is_directory=True)
    rendered = project_map(str(project))
    assert "vendor/" in rendered
    assert "private-notes.txt" not in rendered
