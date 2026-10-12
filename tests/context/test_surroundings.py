"""The context tests do not depend on what surrounds the machine's temporary directory.

``load`` and ``assemble_in`` ask git which repository a directory is in, to read the
instruction files of the directories above it. ``tmp_repo`` has an empty ``.git`` directory,
which git does not take for a repository, so git looks in the directories above, and when the
temporary directory sits inside a work tree (``--basetemp`` set to a directory of a checkout,
a ``TMPDIR`` under a home that is a repository) it answers with that work tree's top. Its
``NANO.md`` or ``CLAUDE.md`` would then be read into tests that built everything they read,
and they would pass or fail with the machine they ran on.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from nanoclaude.context.git import run_git
from tests.context.helpers import assemble_in, load


@pytest.fixture
def a_repository_above_the_temporary_directory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Path]:
    """The directory every test's temporary directory is in, made a repository with a
    ``NANO.md`` of its own, for as long as one test runs."""
    above = tmp_path_factory.getbasetemp()
    subprocess.run(["git", "init", "-q"], cwd=above, check=True)  # noqa: S607
    (above / "NANO.md").write_text(
        "stray rule: from the repository around the temporary directory\n"
    )
    try:
        yield above
    finally:
        shutil.rmtree(above / ".git")
        (above / "NANO.md").unlink()


def test_git_would_answer_with_the_repository_around_the_temporary_directory(
    tmp_repo, a_repository_above_the_temporary_directory
):
    """The hazard is real: asked as it would be on a machine with no ceiling set, git takes
    the work tree above for the one ``tmp_repo`` is in. The tests below are about this."""
    without_a_ceiling = {k: v for k, v in os.environ.items() if k != "GIT_CEILING_DIRECTORIES"}
    asked = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],  # noqa: S607
        cwd=tmp_repo,
        env=without_a_ceiling,
        capture_output=True,
        text=True,
        check=False,
    )
    assert asked.stdout.strip() == str(a_repository_above_the_temporary_directory)


def test_an_instruction_file_around_the_temporary_directory_is_not_read_into_a_test(
    tmp_repo, a_repository_above_the_temporary_directory
):
    (tmp_repo / "NANO.md").write_text("root rule\n")
    loaded = load(tmp_repo)
    assert "root rule" in loaded and "stray rule" not in loaded


def test_nor_into_the_context_a_test_assembles(
    tmp_repo, a_repository_above_the_temporary_directory
):
    (tmp_repo / "NANO.md").write_text("root rule\n")
    context = assemble_in(tmp_repo)
    assert "root rule" in context.system and "stray rule" not in context.system


def test_a_repository_a_test_makes_is_still_found_from_a_directory_inside_it(tmp_path):
    """The ceiling is above everything a test makes: the test's own directory may be the
    repository, and a directory in it is in that repository."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)  # noqa: S607
    inside = tmp_path / "packages" / "app"
    inside.mkdir(parents=True)
    asked = run_git(str(inside), "rev-parse", "--show-toplevel")
    assert asked.returncode == 0 and Path(asked.stdout.strip()) == tmp_path.resolve()
