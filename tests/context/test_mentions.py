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
