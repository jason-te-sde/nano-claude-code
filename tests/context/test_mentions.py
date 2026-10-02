from nanoclaude.context.mentions import expand_mentions
from nanoclaude.permissions.redact import Redactor


def test_a_mentioned_file_is_inlined(tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\n")
    text, paths = expand_mentions("look at @a.py please", root=str(tmp_repo), redactor=Redactor())
    assert "x = 1" in text and paths == (str(tmp_repo / "a.py"),)


def test_a_missing_mention_is_left_as_written(tmp_repo):
    text, paths = expand_mentions("see @nope.py", root=str(tmp_repo), redactor=Redactor())
    assert "@nope.py" in text and paths == ()


def test_a_mention_outside_the_root_is_not_expanded(tmp_repo):
    _, paths = expand_mentions("see @/etc/passwd", root=str(tmp_repo), redactor=Redactor())
    assert paths == ()


def test_secrets_in_a_mentioned_file_are_redacted(tmp_repo):
    (tmp_repo / "cfg.py").write_text('KEY = "AKIAIOSFODNN7EXAMPLE"\n')
    text, _ = expand_mentions("@cfg.py", root=str(tmp_repo), redactor=Redactor())
    assert "AKIA" not in text


def test_an_email_address_is_not_a_mention(tmp_repo):
    text, paths = expand_mentions("mail me at a@b.com", root=str(tmp_repo), redactor=Redactor())
    assert paths == () and "a@b.com" in text


def test_a_secret_path_mention_is_left_as_written_by_default(tmp_repo):
    """Requirement added 2026-10-01 (task-22-brief.md preamble): a mention must
    not inline a secret file. Without this check, only the generic content
    redactor would stand between a mentioned .env and the model -- and that
    redactor matches known credential shapes, not a database URL with an
    inline password, which is exactly what this writes.
    """
    (tmp_repo / ".env").write_text("DATABASE_URL=postgres://u:p@host/db\n")
    text, paths = expand_mentions("@.env", root=str(tmp_repo), redactor=Redactor())
    assert "@.env" in text and paths == ()
    assert "postgres" not in text


def test_a_secret_path_mention_is_expanded_when_allow_secrets_is_set(tmp_repo):
    """The other direction of the same requirement: allow_secrets is an explicit
    opt-in, not a default that happens to be off.
    """
    (tmp_repo / ".env").write_text("hello\n")
    text, paths = expand_mentions(
        "@.env", root=str(tmp_repo), redactor=Redactor(), allow_secrets=True
    )
    assert "hello" in text and paths == (str(tmp_repo / ".env"),)


def test_a_binary_mentioned_file_is_left_as_written(tmp_repo):
    (tmp_repo / "img.bin").write_bytes(b"\x89PNG\x00\x00binary")
    text, paths = expand_mentions("@img.bin", root=str(tmp_repo), redactor=Redactor())
    assert "@img.bin" in text and paths == ()


def test_two_mentions_in_one_message_are_both_inlined(tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\n")
    (tmp_repo / "b.py").write_text("y = 2\n")
    text, paths = expand_mentions(
        "compare @a.py with @b.py", root=str(tmp_repo), redactor=Redactor()
    )
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
    text, paths = expand_mentions("@a.py", root=str(alias), redactor=Redactor())
    assert "x = 1" in text and len(paths) == 1


def test_a_symlink_inside_the_project_to_a_file_outside_is_not_expanded(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("not yours\n")
    (project / "link.txt").symlink_to(outside)
    text, paths = expand_mentions("@link.txt", root=str(project), redactor=Redactor())
    assert paths == () and "not yours" not in text


def test_a_mention_at_the_end_of_a_sentence_ignores_the_full_stop(tmp_repo):
    (tmp_repo / "main.py").write_text("print(1)\n")
    text, paths = expand_mentions("look at @main.py.", root=str(tmp_repo), redactor=Redactor())
    assert "print(1)" in text and len(paths) == 1


def test_mentions_at_the_start_and_the_end_of_the_text_both_expand(tmp_repo):
    (tmp_repo / "a.py").write_text("A = 1\n")
    (tmp_repo / "b.py").write_text("B = 2\n")
    text, paths = expand_mentions(
        "@a.py compare with @b.py", root=str(tmp_repo), redactor=Redactor()
    )
    assert "A = 1" in text and "B = 2" in text and len(paths) == 2


def test_a_file_mentioned_twice_is_inlined_once(tmp_repo):
    (tmp_repo / "a.py").write_text("UNIQUE_MARKER = 1\n")
    text, paths = expand_mentions("@a.py and again @a.py", root=str(tmp_repo), redactor=Redactor())
    assert text.count("UNIQUE_MARKER") == 1 and len(paths) == 1


def test_the_mention_size_cap_counts_bytes_and_says_it_truncated(tmp_repo, monkeypatch):
    import nanoclaude.context.mentions as mentions

    monkeypatch.setattr(mentions, "MAX_MENTION_BYTES", 10)
    (tmp_repo / "wide.txt").write_text("é" * 20)  # two bytes per character
    text, _ = expand_mentions("@wide.txt", root=str(tmp_repo), redactor=Redactor())
    body = text.split('<file path="wide.txt">\n', 1)[1].split("\n[truncated", 1)[0]
    assert len(body.encode("utf-8")) <= 10
    assert "[truncated at 10 bytes]" in text
