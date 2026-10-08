from pathlib import Path

from nanoclaude.context.mentions import expand_mentions
from nanoclaude.permissions.policy import PermissionMode, Policy
from nanoclaude.permissions.redact import SECRET_PATH_PATTERNS, Redactor
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox


def policy_for(
    root: str | Path, *, allow_secrets: bool = False, deny: tuple[str, ...] = ()
) -> Policy:
    return Policy(
        sandbox=Sandbox((str(root),)),
        rules=RuleSet.build(allow=["Read"], deny=list(deny)),
        mode=PermissionMode.DEFAULT,
        secret_paths=SECRET_PATH_PATTERNS,
        allow_secrets=allow_secrets,
    )


def expand(text, *, root, allow_secrets=False, deny=()):
    """``expand_mentions`` as the session calls it: with the policy it runs under."""
    return expand_mentions(
        text,
        root=str(root),
        redactor=Redactor(),
        policy=policy_for(root, allow_secrets=allow_secrets, deny=deny),
    )


def test_a_mentioned_file_is_inlined(tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\n")
    text, paths = expand("look at @a.py please", root=tmp_repo)
    assert "x = 1" in text and paths == (str(tmp_repo / "a.py"),)


def test_a_missing_mention_is_left_as_written(tmp_repo):
    text, paths = expand("see @nope.py", root=tmp_repo)
    assert "@nope.py" in text and paths == ()


def test_a_mention_outside_the_root_is_not_expanded(tmp_repo):
    _, paths = expand("see @/etc/passwd", root=tmp_repo)
    assert paths == ()


def test_secrets_in_a_mentioned_file_are_redacted(tmp_repo):
    (tmp_repo / "cfg.py").write_text('KEY = "AKIAIOSFODNN7EXAMPLE"\n')
    text, _ = expand("@cfg.py", root=tmp_repo)
    assert "AKIA" not in text


def test_an_email_address_is_not_a_mention(tmp_repo):
    text, paths = expand("mail me at a@b.com", root=tmp_repo)
    assert paths == () and "a@b.com" in text


def test_a_secret_path_mention_is_left_as_written_by_default(tmp_repo):
    """Requirement added 2026-10-01 (task-22-brief.md preamble): a mention must
    not inline a secret file. Without this check, only the generic content
    redactor would stand between a mentioned .env and the model -- and that
    redactor matches known credential shapes, not a database URL with an
    inline password, which is exactly what this writes.
    """
    (tmp_repo / ".env").write_text("DATABASE_URL=postgres://u:p@host/db\n")
    text, paths = expand("@.env", root=tmp_repo)
    assert "@.env" in text and paths == ()
    assert "postgres" not in text


def test_a_secret_path_mention_is_expanded_when_allow_secrets_is_set(tmp_repo):
    """The other direction of the same requirement: allow_secrets is an explicit
    opt-in, not a default that happens to be off.
    """
    (tmp_repo / ".env").write_text("hello\n")
    text, paths = expand("@.env", root=tmp_repo, allow_secrets=True)
    assert "hello" in text and paths == (str(tmp_repo / ".env"),)


def test_a_binary_mentioned_file_is_left_as_written(tmp_repo):
    (tmp_repo / "img.bin").write_bytes(b"\x89PNG\x00\x00binary")
    text, paths = expand("@img.bin", root=tmp_repo)
    assert "@img.bin" in text and paths == ()


def test_two_mentions_in_one_message_are_both_inlined(tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\n")
    (tmp_repo / "b.py").write_text("y = 2\n")
    text, paths = expand("compare @a.py with @b.py", root=tmp_repo)
    assert "x = 1" in text and "y = 2" in text
    assert paths == (str(tmp_repo / "a.py"), str(tmp_repo / "b.py"))


def test_mentions_expand_when_the_root_is_reached_through_a_symlink(tmp_path):
    # On macOS everything under /tmp or /var is reached through a symlink; an
    # unresolved root used to make every mention silently fail.
    real = tmp_path / "real"
    real.mkdir()
    (real / "a.py").write_text("x = 1\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    text, paths = expand("@a.py", root=alias)
    assert "x = 1" in text and len(paths) == 1
    assert '<file path="a.py">' in text  # shown relative to the resolved root


def test_a_symlink_inside_the_project_to_a_file_outside_is_not_expanded(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not yours\n")
    (project / "link.txt").symlink_to(outside)
    text, paths = expand("@link.txt", root=project)
    assert paths == () and "not yours" not in text


def test_a_mention_at_the_end_of_a_sentence_ignores_the_full_stop(tmp_repo):
    (tmp_repo / "main.py").write_text("print(1)\n")
    text, paths = expand("look at @main.py.", root=tmp_repo)
    assert "print(1)" in text and len(paths) == 1


def test_mentions_at_the_start_and_the_end_of_the_text_both_expand(tmp_repo):
    (tmp_repo / "a.py").write_text("A = 1\n")
    (tmp_repo / "b.py").write_text("B = 2\n")
    text, paths = expand("@a.py compare with @b.py", root=tmp_repo)
    assert "A = 1" in text and "B = 2" in text and len(paths) == 2


def test_a_file_mentioned_twice_is_inlined_once(tmp_repo):
    (tmp_repo / "a.py").write_text("UNIQUE_MARKER = 1\n")
    text, paths = expand("@a.py and again @a.py", root=tmp_repo)
    assert text.count("UNIQUE_MARKER") == 1 and len(paths) == 1


def test_the_mention_size_cap_counts_bytes_and_says_it_truncated(tmp_repo, monkeypatch):
    import nanoclaude.context.mentions as mentions

    monkeypatch.setattr(mentions, "MAX_MENTION_BYTES", 10)
    (tmp_repo / "wide.txt").write_text("é" * 20)  # two bytes per character
    text, _ = expand("@wide.txt", root=tmp_repo)
    body = text.split('<file path="wide.txt">\n', 1)[1].split("\n[truncated", 1)[0]
    assert len(body.encode("utf-8")) <= 10
    assert "[truncated at 10 bytes]" in text


def test_a_credential_straddling_the_size_cap_is_scrubbed_before_the_cut(tmp_repo, monkeypatch):
    import nanoclaude.context.mentions as mentions

    monkeypatch.setattr(mentions, "MAX_MENTION_BYTES", 12)
    (tmp_repo / "cfg.txt").write_text("xxxx " + "AKIAIOSFODNN7EXAMPLE")
    text, _ = expand("@cfg.txt", root=tmp_repo)
    assert "AKIA" not in text


def test_a_relative_root_is_refused():
    import pytest

    with pytest.raises(ValueError, match="absolute"):
        expand_mentions("@a.py", root="relative/dir", redactor=Redactor(), policy=policy_for("/p"))


def test_a_file_a_read_deny_rule_covers_is_not_inlined_and_the_text_says_why(tmp_repo):
    (tmp_repo / "secrets").mkdir()
    (tmp_repo / "secrets" / "notes.txt").write_text("launch code 1234\n")
    text, paths = expand("explain @secrets/notes.txt", root=tmp_repo, deny=("Read(secrets/**)",))
    assert paths == ()
    assert "launch code" not in text and "<file" not in text
    assert "@secrets/notes.txt" in text
    notes = [line for line in text.splitlines() if "not inlined" in line]
    assert len(notes) == 1, text
    assert "secrets/notes.txt" in notes[0]
    assert "rule.deny" in notes[0] and "Read(secrets/**)" in notes[0]


def test_a_denied_file_mentioned_twice_is_explained_once(tmp_repo):
    (tmp_repo / "secrets").mkdir()
    (tmp_repo / "secrets" / "notes.txt").write_text("launch code 1234\n")
    text, _ = expand(
        "@secrets/notes.txt and again @secrets/notes.txt",
        root=tmp_repo,
        deny=("Read(secrets/**)",),
    )
    assert text.count("not inlined") == 1


def test_the_other_files_in_the_message_are_still_inlined_around_a_denied_one(tmp_repo):
    (tmp_repo / "secrets").mkdir()
    (tmp_repo / "secrets" / "notes.txt").write_text("launch code 1234\n")
    (tmp_repo / "a.py").write_text("x = 1\n")
    (tmp_repo / "b.py").write_text("y = 2\n")
    text, paths = expand(
        "@a.py then @secrets/notes.txt then @b.py", root=tmp_repo, deny=("Read(secrets/**)",)
    )
    assert "x = 1" in text and "y = 2" in text and "launch code" not in text
    assert paths == (str(tmp_repo / "a.py"), str(tmp_repo / "b.py"))
    assert text.index("x = 1") < text.index("not inlined") < text.index("y = 2")


def test_a_credentials_path_mention_says_why_it_was_not_inlined(tmp_repo):
    (tmp_repo / ".env").write_text("DATABASE_URL=postgres://u:p@host/db\n")
    text, paths = expand("@.env", root=tmp_repo)
    assert paths == () and "postgres" not in text
    notes = [line for line in text.splitlines() if "not inlined" in line]
    assert len(notes) == 1 and "secret.path" in notes[0], text


def test_a_deny_rule_for_another_tool_does_not_stop_a_mention(tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\n")
    text, paths = expand("@a.py", root=tmp_repo, deny=("Write(a.py)",))
    assert "x = 1" in text and len(paths) == 1 and "not inlined" not in text


def test_a_link_to_a_denied_file_is_not_inlined_under_the_links_name(tmp_repo):
    (tmp_repo / "secrets").mkdir()
    (tmp_repo / "secrets" / "notes.txt").write_text("launch code 1234\n")
    (tmp_repo / "link.txt").symlink_to(tmp_repo / "secrets" / "notes.txt")
    text, paths = expand("@link.txt", root=tmp_repo, deny=("Read(secrets/**)",))
    assert paths == () and "launch code" not in text


def test_a_link_named_like_a_credentials_file_is_not_inlined(tmp_repo):
    (tmp_repo / "harmless.txt").write_text("fine\n")
    (tmp_repo / ".env").symlink_to(tmp_repo / "harmless.txt")
    text, paths = expand("@.env", root=tmp_repo)
    assert paths == () and "fine" not in text


def test_the_note_is_one_line_however_odd_the_path_it_names(tmp_repo):
    """The refusal quotes the path the mention resolved to, and a name can hold a newline
    or a terminal escape."""
    odd = tmp_repo / "odd\ndir\x1b[2J"
    odd.mkdir()
    (odd / ".env").write_text("x\n")
    (tmp_repo / "link.txt").symlink_to(odd / ".env")
    text, paths = expand("@link.txt", root=tmp_repo)
    assert paths == ()
    (note,) = [line for line in text.splitlines() if "not inlined" in line]
    assert note.startswith("[") and note.endswith("]")
    assert "\x1b" not in text
