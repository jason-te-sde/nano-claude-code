import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.context.projectmap import project_map
from tests.context.helpers import policy_for


def mapped(
    root: Path, *, deny: tuple[str, ...] = (), allow_secrets: bool = False, **kw: Any
) -> str:
    """The map of ``root`` as a session whose policy is the default one, plus ``deny`` rules."""
    policy = replace(policy_for(root, deny=deny), allow_secrets=allow_secrets)
    return project_map(str(root), policy=policy, **kw)


def entries(rendered: str) -> set[str]:
    return {line.strip() for line in rendered.splitlines()}


def test_the_map_shows_the_tree_and_respects_gitignore(tmp_repo):
    (tmp_repo / "src").mkdir()
    (tmp_repo / "src" / "main.py").write_text("")
    (tmp_repo / "node_modules").mkdir()
    (tmp_repo / "node_modules" / "junk.js").write_text("")
    rendered = mapped(tmp_repo)
    assert "src/" in rendered and "main.py" in rendered
    assert "node_modules" not in rendered


def test_depth_is_limited(tmp_repo):
    deep = tmp_repo / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "deep.py").write_text("")
    rendered = mapped(tmp_repo, depth=2)
    assert "deep.py" not in rendered


def test_a_symlink_loop_does_not_hang(tmp_repo):
    """Review Focus #4. Ordinary in real repositories, fatal if walked naively."""
    (tmp_repo / "sub").mkdir()
    (tmp_repo / "sub" / "loop").symlink_to(tmp_repo)
    rendered = mapped(tmp_repo, depth=5)
    assert isinstance(rendered, str)


def test_the_entry_count_is_capped_and_says_so(tmp_repo):
    for index in range(500):
        (tmp_repo / f"f{index}.py").write_text("")
    rendered = mapped(tmp_repo, max_entries=50)
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
    rendered = mapped(tmp_path)
    assert "a.py" in rendered and ".git" not in rendered


def test_a_root_that_does_not_exist_does_not_crash(tmp_path):
    """directory.stat() in walk()'s very first call -- on the root itself, with
    no prior is_dir() check from a parent iteration to have already ruled this
    out -- can fail just as surely as it can one level down. A non-existent
    root is the simplest deterministic way to reach that except branch.
    """
    rendered = mapped(tmp_path / "does-not-exist")
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
        rendered = mapped(tmp_repo)
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
    rendered = mapped(project)
    assert "vendor/" in rendered
    assert "private-notes.txt" not in rendered


def test_an_in_project_link_is_entered_when_the_root_is_reached_through_a_symlink(tmp_path):
    # The target sits beyond the depth limit, so the link is the only way the
    # walk can reach found.txt -- which makes the containment check decide it.
    real = tmp_path / "real"
    deep = real / "x" / "y" / "z"
    deep.mkdir(parents=True)
    (deep / "found.txt").write_text("x")
    (real / "a_link").symlink_to(deep, target_is_directory=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    assert "found.txt" in mapped(alias, depth=2)


# ---- what the policy would not let Read show is not in the map


CREDENTIALS = [
    ".env",
    ".env.local",
    ".ENV",
    "prod.env",
    "id_rsa",
    "credentials",
    "credentials.json",
    "secrets.yaml",
    "server.pem",
    "deploy.key",
    ".npmrc",
    ".netrc",
    "terraform.tfstate",
    ".htpasswd",
]


def test_the_names_of_credentials_files_are_not_in_the_map(tmp_repo):
    for name in CREDENTIALS:
        (tmp_repo / name).write_text("x")
    (tmp_repo / "src").mkdir()
    (tmp_repo / "src" / "a.py").write_text("")
    (tmp_repo / "src" / "app.env").write_text("x")
    (tmp_repo / "README.md").write_text("")
    shown = entries(mapped(tmp_repo))
    assert {"src/", "a.py", "README.md"} <= shown
    assert shown.isdisjoint({*CREDENTIALS, "app.env"}), shown


def test_the_files_inside_a_credentials_directory_are_not_in_the_map(tmp_repo):
    (tmp_repo / ".aws").mkdir()
    (tmp_repo / ".aws" / "config").write_text("x")
    (tmp_repo / ".ssh").mkdir()
    (tmp_repo / ".ssh" / "known_hosts").write_text("x")
    (tmp_repo / "deploy" / ".gnupg").mkdir(parents=True)
    (tmp_repo / "deploy" / ".gnupg" / "trustdb.gpg").write_text("x")
    (tmp_repo / "deploy" / "run.sh").write_text("x")
    shown = entries(mapped(tmp_repo))
    assert "run.sh" in shown
    assert shown.isdisjoint({"config", "known_hosts", "trustdb.gpg"}), shown


def test_what_a_read_deny_rule_covers_is_not_in_the_map(tmp_repo):
    (tmp_repo / "hidden").mkdir()
    (tmp_repo / "hidden" / "plans.txt").write_text("x")
    (tmp_repo / "vault").mkdir()
    (tmp_repo / "vault" / "inner.txt").write_text("x")
    (tmp_repo / "notes.txt").write_text("x")
    (tmp_repo / "open.txt").write_text("x")
    shown = entries(mapped(tmp_repo, deny=("Read(hidden/**)", "Read(vault)", "Read(notes.txt)")))
    assert "open.txt" in shown
    assert shown.isdisjoint({"plans.txt", "vault/", "inner.txt", "notes.txt"}), shown


def test_a_deny_rule_for_another_tool_does_not_hide_anything_from_the_map(tmp_repo):
    (tmp_repo / "notes.txt").write_text("x")
    assert "notes.txt" in entries(mapped(tmp_repo, deny=("Write(notes.txt)",)))


def test_a_refusal_is_by_letter_case_too(tmp_repo):
    (tmp_repo / "Secrets").mkdir()
    (tmp_repo / "Secrets" / "k.txt").write_text("x")
    (tmp_repo / ".Env").write_text("x")
    shown = entries(mapped(tmp_repo, deny=("Read(secrets/**)",)))
    assert shown.isdisjoint({"k.txt", ".Env"}), shown


def test_allowing_secrets_puts_the_credentials_files_back_in_the_map(tmp_repo):
    (tmp_repo / ".env").write_text("x")
    assert ".env" in entries(mapped(tmp_repo, allow_secrets=True))
    assert ".env" not in entries(mapped(tmp_repo))


def test_a_link_to_a_credentials_file_is_judged_by_where_it_leads_as_well(tmp_repo):
    (tmp_repo / ".env").write_text("x")
    (tmp_repo / "notes").symlink_to(tmp_repo / ".env")
    (tmp_repo / "real.txt").write_text("x")
    (tmp_repo / "alias.txt").symlink_to(tmp_repo / "real.txt")
    shown = entries(mapped(tmp_repo))
    assert "notes" not in shown and ".env" not in shown
    assert {"real.txt", "alias.txt"} <= shown


def test_a_link_named_like_a_credentials_file_is_judged_by_its_name_too(tmp_repo):
    (tmp_repo / "harmless.txt").write_text("x")
    (tmp_repo / ".env").symlink_to(tmp_repo / "harmless.txt")
    shown = entries(mapped(tmp_repo))
    assert ".env" not in shown and "harmless.txt" in shown


def test_a_link_to_a_directory_a_deny_rule_covers_does_not_list_what_is_in_it(tmp_repo):
    (tmp_repo / "secrets").mkdir()
    (tmp_repo / "secrets" / "key.txt").write_text("x")
    (tmp_repo / "data").symlink_to(tmp_repo / "secrets", target_is_directory=True)
    shown = entries(mapped(tmp_repo, deny=("Read(secrets/**)",)))
    assert "key.txt" not in shown


def test_a_refused_entry_does_not_use_up_the_entry_cap(tmp_repo):
    for index in range(60):
        (tmp_repo / f"k{index}.env").write_text("x")
    for index in range(10):
        (tmp_repo / f"f{index}.py").write_text("")
    # the fixture's .gitignore is an entry too: eleven in all, and sixty refused ones after them
    rendered = mapped(tmp_repo, max_entries=11)
    assert "truncated" not in rendered, rendered
    assert sum(name.endswith(".py") for name in entries(rendered)) == 10
    assert len(entries(mapped(tmp_repo, max_entries=5))) == 5 + 1  # five, and the note


def test_the_assembled_context_leaves_credentials_names_out_of_the_map(tmp_repo):
    from tests.context.helpers import assemble_in

    (tmp_repo / ".env").write_text("x")
    (tmp_repo / "app.py").write_text("")
    context = assemble_in(tmp_repo)
    assert "app.py" in context.environment
    assert ".env" not in context.environment
