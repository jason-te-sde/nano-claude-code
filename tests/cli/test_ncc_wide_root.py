"""Tests for a sandbox that would be a whole home directory, or the whole machine, by accident.

``ncc`` run with no ``--root`` in ``$HOME`` made all of it the sandbox, the session store and
the shell's startup files with it. Run that way, ncc stops with a line in the form of spec
17.9 and exit 2 before anything is opened. Naming the directory with ``--root`` says it was
meant.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanoclaude.cli.main import EXIT_CODES
from nanoclaude.testing.scripted import calls, says
from tests.cli.helpers import run_ncc

HOME_LINE = (
    "error: the current directory is your home directory — every file in it would be in "
    "reach; run ncc from a project directory, or pass --root ~ if you mean it\n"
)
ROOT_LINE = (
    "error: the current directory is the filesystem root — every file on this machine would "
    "be in reach; run ncc from a project directory, or pass --root / if you mean it\n"
)


def refused(capsys: pytest.CaptureFixture[str], *argv: str) -> str:
    """Run ncc, which must stop as a usage error with nothing on stdout; its stderr."""
    code, out, err = run_ncc(capsys, *argv)
    assert (code, out) == (EXIT_CODES["usage"], ""), (code, out, err)
    return err


@pytest.mark.parametrize("argv", [["-p", "hi"], []], ids=["headless", "interactive"])
def test_ncc_run_in_the_home_directory_without_a_root_is_refused(
    ncc_home, serve, capsys, monkeypatch, argv
):
    clients = serve(m=[says("never asked")])
    monkeypatch.chdir(ncc_home)
    assert refused(capsys, *argv) == HOME_LINE
    assert clients["m"].requests == []
    assert not (ncc_home / ".nanoclaude" / "sessions.db").exists()


def test_a_directory_that_is_a_link_to_the_home_directory_is_the_home_directory(
    ncc_home, serve, capsys, monkeypatch, tmp_path
):
    serve(m=[says("never asked")])
    (tmp_path / "shortcut").symlink_to(ncc_home, target_is_directory=True)
    monkeypatch.chdir(tmp_path / "shortcut")
    assert refused(capsys, "-p", "hi") == HOME_LINE


def test_ncc_run_in_the_filesystem_root_without_a_root_is_refused(
    ncc_home, serve, capsys, monkeypatch
):
    clients = serve(m=[says("never asked")])
    monkeypatch.chdir("/")
    assert refused(capsys, "-p", "hi") == ROOT_LINE
    assert clients["m"].requests == []


def test_the_real_home_is_refused_when_ncc_keeps_its_files_elsewhere(
    ncc_home, serve, capsys, monkeypatch, tmp_path
):
    """NANOCLAUDE_HOME moves where ncc keeps its files and does not move the person's home,
    whose startup files are what the sandbox would hold."""
    serve(m=[says("never asked")])
    real_home = tmp_path / "the-persons-home"
    real_home.mkdir()
    monkeypatch.setenv("HOME", str(real_home))
    monkeypatch.chdir(real_home)
    assert refused(capsys, "-p", "hi") == HOME_LINE


def test_the_directory_ncc_keeps_its_files_in_is_refused_too(
    ncc_home, serve, capsys, monkeypatch, tmp_path
):
    serve(m=[says("never asked")])
    other_home = tmp_path / "somewhere-else"
    other_home.mkdir()
    monkeypatch.setenv("HOME", str(other_home))
    monkeypatch.chdir(ncc_home)  # NANOCLAUDE_HOME
    assert refused(capsys, "-p", "hi") == HOME_LINE


def test_a_directory_that_holds_the_home_directory_is_refused_and_says_so(
    ncc_home, serve, capsys, monkeypatch
):
    """The directory above home has all of it in it, the session store and startup files too."""
    serve(m=[says("never asked")])
    monkeypatch.chdir(ncc_home.parent)
    assert refused(capsys, "-p", "hi") == (
        "error: the current directory contains your home directory — every file in it would be "
        f"in reach; run ncc from a project directory, or pass --root {ncc_home.parent} if you "
        "mean it\n"
    )


def test_a_project_inside_the_home_directory_is_fine(ncc_home, serve, capsys, monkeypatch):
    project = ncc_home / "src" / "app"
    project.mkdir(parents=True)
    (project / "a.txt").write_text("here\n")
    monkeypatch.chdir(project)
    clients = serve(m=[calls("Read", {"path": "a.txt"}, call_id="r1"), says("done")])
    code, _, _ = run_ncc(capsys, "-p", "read it")
    assert code == EXIT_CODES["completed"]
    assert any("here" in b.content for b in clients["m"].requests[1].transcript.messages[-1].blocks)


@pytest.mark.parametrize("root", ["~", ".", "$HOME"], ids=["tilde", "dot", "path"])
def test_naming_the_home_directory_with_root_says_it_was_meant(
    ncc_home, serve, capsys, monkeypatch, root
):
    serve(m=[says("done")])
    monkeypatch.chdir(ncc_home)
    named = str(ncc_home) if root == "$HOME" else root
    code, out, err = run_ncc(capsys, "--root", named, "-p", "hi")
    assert (code, out, err) == (EXIT_CODES["completed"], "done\n", "")


def test_naming_the_filesystem_root_with_root_says_it_was_meant(
    ncc_home, serve, capsys, monkeypatch, sessions
):
    serve(m=[says("done")])
    monkeypatch.chdir("/")
    code, _, err = run_ncc(capsys, "--root", "/", "-p", "hi")
    assert (code, err) == (EXIT_CODES["completed"], "")
    assert sessions[0].root == "/"


def test_an_added_directory_is_not_a_root_that_was_not_named(
    ncc_home, project, serve, capsys, monkeypatch
):
    """--add-dir is an explicit act, and the root it adds to is the project's."""
    serve(m=[says("done")])
    monkeypatch.chdir(project)
    code, _, err = run_ncc(capsys, "--add-dir", "~", "-p", "hi")
    assert (code, err) == (EXIT_CODES["completed"], "")


@pytest.mark.parametrize("variable", ["HOME", "NANOCLAUDE_HOME"])
def test_a_home_that_is_named_through_a_link_is_the_home_the_directory_is(
    ncc_home, serve, capsys, monkeypatch, tmp_path, variable
):
    """Each of the two homes is compared as the real path it is: the other one is somewhere
    else, so that it cannot be the one that catches it."""
    serve(m=[says("never asked")])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "home-link").symlink_to(ncc_home, target_is_directory=True)
    monkeypatch.setenv("HOME", str(elsewhere))
    monkeypatch.setenv("NANOCLAUDE_HOME", str(elsewhere))
    monkeypatch.setenv(variable, str(tmp_path / "home-link"))
    monkeypatch.chdir(ncc_home)
    assert refused(capsys, "-p", "hi") == HOME_LINE


def test_a_machine_with_no_findable_home_still_runs_in_a_project(
    ncc_home, project, serve, capsys, monkeypatch
):
    """Path.home() raises when there is no HOME and no entry for the user; the refusal is
    then only for the home ncc itself keeps its files in."""

    def nobody(_cls: object) -> Path:
        raise RuntimeError("Could not determine home directory.")

    serve(m=[says("done")])
    monkeypatch.setattr(Path, "home", classmethod(nobody))
    monkeypatch.chdir(project)
    code, out, err = run_ncc(capsys, "-p", "hi")
    assert (code, out, err) == (EXIT_CODES["completed"], "done\n", "")
