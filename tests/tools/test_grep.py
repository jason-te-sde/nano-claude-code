import shutil
from dataclasses import replace
from typing import Never

import pytest

from nanoclaude.tools.grep import GrepTool

EXPECTED_DESCRIPTION = """Search file contents with a regular expression.

- `pattern` is a regular expression (ripgrep syntax), not a shell glob.
- `glob` limits which files are searched, for example `*.py`.
- `output_mode` is `content` (matching lines, the default), `files` (paths only) or
  `count` (matches per file).
- `-A`, `-B` and `-C` add lines of context; `-i` is case-insensitive.
- Paths ignored by .gitignore are skipped. Use `head_limit` to cap the output."""

#: Both search backends must agree on every behavior pinned in this file. The
#: "ripgrep" case is skipped where rg is not on PATH; "python-fallback" forces
#: the pure-Python path explicitly (rather than only running it incidentally
#: on a machine that happens to lack rg), per the task's own requirement that
#: both backends -- not just whichever one the host happens to have -- are
#: exercised.
FORCE_PYTHON_FALLBACK = [
    pytest.param(
        False,
        marks=pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not on PATH"),
        id="ripgrep",
    ),
    pytest.param(True, id="python-fallback"),
]


def _maybe_force_python_fallback(
    monkeypatch: pytest.MonkeyPatch, force_python_fallback: bool
) -> None:
    if force_python_fallback:
        monkeypatch.setattr("nanoclaude.tools.search.ripgrep_available", lambda: False)


async def test_content_mode_shows_path_line_and_text(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("def hello():\n    pass\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "def hello"})
    assert "a.py:1:" in outcome.content and "def hello" in outcome.content


async def test_files_mode_lists_paths_only(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("hello\nhello\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "hello", "output_mode": "files"})
    assert outcome.content.count("a.py") == 1


async def test_count_mode_reports_per_file_totals(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("hello\nhello\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "hello", "output_mode": "count"})
    assert "2" in outcome.content


async def test_an_invalid_regex_is_an_error_the_model_can_fix(ctx):
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "("})
    assert outcome.is_error and "regular expression" in outcome.content


async def test_secrets_in_matched_lines_are_redacted(ctx, tmp_repo):
    (tmp_repo / "config.py").write_text('TOKEN = "' + "ghp_" + "A" * 36 + '"\n')
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "TOKEN"})
    assert "ghp_" not in outcome.content


def test_grep_is_read_only_and_needs_no_paths_resolved(ctx):
    request = GrepTool().permission_request(ctx, {"pattern": "hello"})
    assert GrepTool().read_only and request.is_write is False


def test_the_permission_request_resolves_a_path_argument_when_given(ctx, tmp_repo):
    request = GrepTool().permission_request(ctx, {"pattern": "hello", "path": "sub"})
    assert request.resolved_paths == (ctx.resolve("sub"),)


async def test_a_path_argument_limits_the_search_to_a_subdirectory(ctx, tmp_repo):
    (tmp_repo / "sub").mkdir()
    (tmp_repo / "sub" / "a.py").write_text("hello\n")
    (tmp_repo / "top.py").write_text("hello\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "hello", "path": "sub"})
    assert "a.py" in outcome.content
    assert "top.py" not in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_the_glob_argument_restricts_which_files_are_searched(
    ctx, tmp_repo, monkeypatch, force_python_fallback
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / "a.py").write_text("hello\n")
    (tmp_repo / "a.txt").write_text("hello\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "hello", "glob": "*.py"})
    assert "a.py" in outcome.content
    assert "a.txt" not in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_dash_i_makes_the_search_case_insensitive(
    ctx, tmp_repo, monkeypatch, force_python_fallback
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / "a.py").write_text("HELLO\n")
    insensitive = await GrepTool().run(ctx, "t1", {"pattern": "hello", "-i": True})
    assert not insensitive.is_error and "HELLO" in insensitive.content
    sensitive = await GrepTool().run(ctx, "t1", {"pattern": "hello"})
    assert "No matches for" in sensitive.content


async def test_head_limit_caps_how_many_matches_come_back(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("hello\n" * 10)
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "hello", "head_limit": 3})
    assert "3 match(es)" in outcome.content


async def test_no_matches_says_so(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("nothing interesting\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "zzz_not_present"})
    assert not outcome.is_error
    assert "No matches for" in outcome.content


async def test_a_search_failure_is_an_error_result_not_an_exception(ctx, tmp_repo, monkeypatch):
    def boom(*_args: object, **_kwargs: object) -> Never:
        raise RuntimeError("ripgrep exploded")

    monkeypatch.setattr("nanoclaude.tools.grep.search", boom)
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "hello"})
    assert outcome.is_error and "search failed" in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_secret_files_are_dropped_from_matches_by_default(
    ctx, tmp_repo, monkeypatch, force_python_fallback
):
    """Spec Sec.4's threat model: a broad search must not surface an un-ignored
    .env's contents just because the model never named the file directly. The
    line below is deliberately not shaped like anything _SHAPES or _ASSIGNED
    (permissions/redact.py) would catch on its own -- this is pinning the
    file-level filter, not the content-level one, which is covered separately
    by test_secrets_in_matched_lines_are_redacted above.
    """
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / ".env").write_text("DATABASE_URL=postgres://admin:hunter2@db.internal:5432/prod\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "DATABASE_URL"})
    assert not outcome.is_error
    assert ".env" not in outcome.content
    assert "hunter2" not in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_allow_secrets_lets_secret_file_matches_through(
    ctx, tmp_repo, monkeypatch, force_python_fallback
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / ".env").write_text("DATABASE_URL=postgres://admin:hunter2@db.internal:5432/prod\n")
    allowed_ctx = replace(ctx, policy=replace(ctx.policy, allow_secrets=True))
    outcome = await GrepTool().run(allowed_ctx, "t1", {"pattern": "DATABASE_URL"})
    assert not outcome.is_error
    assert ".env" in outcome.content
    assert "hunter2" in outcome.content


def test_the_description_is_byte_identical_to_the_spec():
    """It is a prompt. Changing a word needs a design note, so pin the shape."""
    assert GrepTool().spec().description == EXPECTED_DESCRIPTION
