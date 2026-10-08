import shutil
from dataclasses import replace
from pathlib import Path
from typing import Never

import pytest

from nanoclaude.permissions.rules import RuleSet
from nanoclaude.tools.base import ToolArgumentError, ToolContext
from nanoclaude.tools.grep import GrepTool
from nanoclaude.tools.search import Match

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


def test_permission_request_raises_on_a_missing_pattern(ctx):
    """See test_glob.py's equivalent: a malformed call must raise here, not be
    tolerated."""
    with pytest.raises(ToolArgumentError, match="pattern must be a string"):
        GrepTool().permission_request(ctx, {})


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


async def test_no_separator_between_far_apart_matches_without_context(ctx, tmp_repo):
    """Without -A/-B/-C, two matches many lines apart are each their own
    result, not two "groups" -- no `--` belongs between them, however far
    apart they are (see _format_content's `merging` gate).
    """
    (tmp_repo / "a.py").write_text("MATCH_A\n" + "filler\n" * 10 + "MATCH_B\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH"})
    assert "--" not in outcome.content.splitlines()
    assert "a.py:1:MATCH_A" in outcome.content
    assert "a.py:12:MATCH_B" in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_dash_a_shows_only_after_context(ctx, tmp_repo, monkeypatch, force_python_fallback):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / "a.py").write_text("before\nMATCH\nafter1\nafter2\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-A": 2})
    assert "a.py:2:MATCH" in outcome.content
    assert "a.py-3-after1" in outcome.content
    assert "a.py-4-after2" in outcome.content
    assert "before" not in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_dash_b_shows_only_before_context(ctx, tmp_repo, monkeypatch, force_python_fallback):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / "a.py").write_text("before1\nbefore2\nMATCH\nafter\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-B": 2})
    assert "a.py-1-before1" in outcome.content
    assert "a.py-2-before2" in outcome.content
    assert "a.py:3:MATCH" in outcome.content
    assert "after" not in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_dash_c_sets_both_sides(ctx, tmp_repo, monkeypatch, force_python_fallback):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / "a.py").write_text("before\nMATCH\nafter\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-C": 1})
    assert "a.py-1-before" in outcome.content
    assert "a.py:2:MATCH" in outcome.content
    assert "a.py-3-after" in outcome.content


async def test_explicit_dash_b_overrides_dash_c_on_that_side_only(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("before\nMATCH\nafter1\nafter2\nafter3\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-C": 3, "-B": 0})
    assert "before" not in outcome.content
    assert "a.py:2:MATCH" in outcome.content
    assert "a.py-3-after1" in outcome.content
    assert "a.py-4-after2" in outcome.content
    assert "a.py-5-after3" in outcome.content


async def test_explicit_dash_a_overrides_dash_c_on_that_side_only(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("before1\nbefore2\nbefore3\nMATCH\nafter\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-C": 3, "-A": 0})
    assert "after" not in outcome.content
    assert "a.py-1-before1" in outcome.content
    assert "a.py-2-before2" in outcome.content
    assert "a.py-3-before3" in outcome.content
    assert "a.py:4:MATCH" in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_non_adjacent_groups_get_a_separator(
    ctx, tmp_repo, monkeypatch, force_python_fallback
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / "a.py").write_text("MATCH_A\n" + "filler\n" * 10 + "MATCH_B\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-C": 1})
    assert "--" in outcome.content.splitlines()
    assert "a.py:1:MATCH_A" in outcome.content
    assert "a.py:12:MATCH_B" in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_touching_groups_get_no_separator(ctx, tmp_repo, monkeypatch, force_python_fallback):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / "a.py").write_text("MATCH_A\nmiddle\nMATCH_B\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-C": 1})
    assert "--" not in outcome.content.splitlines()


async def test_files_mode_ignores_context(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("before\nMATCH\nafter\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-C": 1, "output_mode": "files"})
    assert outcome.content.count("a.py") == 1
    assert "before" not in outcome.content
    assert "after" not in outcome.content


async def test_count_mode_ignores_context(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("before\nMATCH\nafter\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-C": 1, "output_mode": "count"})
    assert "a.py: 1" in outcome.content


async def test_head_limit_counts_matches_not_context_lines(ctx, tmp_repo):
    """head_limit still counts matches (see search.py's _apply_match_limit):
    a limit of 1 does not cut a single match's own surrounding context short,
    it only stops a *second* match's group from starting.
    """
    (tmp_repo / "a.py").write_text("MATCH_A\nl2\nl3\nl4\nl5\nMATCH_B\nl7\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "MATCH", "-C": 1, "head_limit": 1})
    assert "1 match(es)" in outcome.content
    assert "a.py-2-l2" in outcome.content
    assert "MATCH_B" not in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_context_lines_never_come_from_a_secret_file(
    ctx, tmp_repo, monkeypatch, force_python_fallback
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / ".env").write_text(
        "before\nDATABASE_URL=postgres://admin:hunter2@db.internal:5432/prod\nafter\n"
    )
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "DATABASE_URL", "-C": 1})
    assert not outcome.is_error
    assert ".env" not in outcome.content
    assert "hunter2" not in outcome.content
    assert "before" not in outcome.content
    assert "after" not in outcome.content


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


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_a_file_symlink_to_outside_the_sandbox_surfaces_nothing(
    layout, monkeypatch, force_python_fallback
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (layout.outside / "private.txt").write_text("needle from outside\n")
    (layout.project / "notes.txt").symlink_to(layout.outside / "private.txt")
    (layout.project / "real.txt").write_text("needle from inside\n")
    for mode in ("content", "files", "count"):
        outcome = await GrepTool().run(layout.ctx, "t1", {"pattern": "needle", "output_mode": mode})
        assert "from outside" not in outcome.content, f"{mode}: {outcome.content}"
        assert "notes.txt" not in outcome.content, f"{mode}: {outcome.content}"
        assert "real.txt" in outcome.content, f"{mode}: {outcome.content}"


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_a_directory_symlink_to_outside_the_sandbox_surfaces_nothing(
    layout, monkeypatch, force_python_fallback
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (layout.outside / "private.txt").write_text("needle from outside\n")
    (layout.project / "vendor").symlink_to(layout.outside, target_is_directory=True)
    (layout.project / "real.txt").write_text("needle from inside\n")
    outcome = await GrepTool().run(layout.ctx, "t1", {"pattern": "needle"})
    assert "from outside" not in outcome.content
    assert "vendor" not in outcome.content
    assert "real.txt" in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_a_link_inside_the_sandbox_to_a_credentials_file_surfaces_nothing(
    layout, monkeypatch, force_python_fallback
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (layout.project / ".env").write_text("needle=postgres://admin:hunter2@db.internal/prod\n")
    (layout.project / "notes.txt").symlink_to(layout.project / ".env")
    outcome = await GrepTool().run(layout.ctx, "t1", {"pattern": "needle", "-C": 1})
    assert "hunter2" not in outcome.content
    assert "notes.txt" not in outcome.content


def _backend_that_reports(monkeypatch: pytest.MonkeyPatch, *paths: Path) -> None:
    """A search backend that follows links: it reports a match in each of ``paths``
    as it is spelled, whatever those paths lead to."""
    monkeypatch.setattr(
        "nanoclaude.tools.grep.search",
        lambda *_a, **_k: [Match(str(p), 1, f"needle in {p.name}") for p in paths],
    )


async def test_matches_are_judged_by_where_their_path_leads_whichever_backend_found_them(
    layout, monkeypatch
):
    """Grep does not trust a backend to have skipped what it must not show: a path
    that resolves outside the sandbox, or to a credentials file, is dropped here
    on what it resolves to, however innocent its own name is."""
    (layout.outside / "private.txt").write_text("outside\n")
    (layout.project / ".env").write_text("secret\n")
    (layout.project / "to_outside.txt").symlink_to(layout.outside / "private.txt")
    (layout.project / "to_env.txt").symlink_to(layout.project / ".env")
    (layout.project / "real.txt").write_text("fine\n")
    _backend_that_reports(
        monkeypatch,
        layout.project / "to_outside.txt",
        layout.project / "to_env.txt",
        layout.project / "real.txt",
    )
    outcome = await GrepTool().run(layout.ctx, "t1", {"pattern": "needle"})
    assert "real.txt" in outcome.content
    assert "to_outside" not in outcome.content
    assert "to_env" not in outcome.content


async def test_a_link_that_leads_somewhere_readable_inside_the_sandbox_is_kept(layout, monkeypatch):
    """The check must not alarm on everything: a link to an ordinary file in the project
    is as readable as the file."""
    (layout.project / "real.txt").write_text("fine\n")
    (layout.project / "alias.txt").symlink_to(layout.project / "real.txt")
    _backend_that_reports(monkeypatch, layout.project / "alias.txt")
    outcome = await GrepTool().run(layout.ctx, "t1", {"pattern": "needle"})
    assert "needle in alias.txt" in outcome.content


async def test_a_link_named_like_a_credentials_file_is_dropped_by_its_name(layout, monkeypatch):
    (layout.project / "harmless.txt").write_text("fine\n")
    (layout.project / ".env").symlink_to(layout.project / "harmless.txt")
    _backend_that_reports(monkeypatch, layout.project / ".env")
    outcome = await GrepTool().run(layout.ctx, "t1", {"pattern": "needle"})
    assert ".env" not in outcome.content


def _deny_reading(ctx: ToolContext, *rules: str) -> ToolContext:
    return replace(ctx, policy=replace(ctx.policy, rules=RuleSet.build(deny=list(rules))))


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
async def test_a_read_deny_rule_also_keeps_grep_out_of_the_files_it_covers(
    ctx, tmp_repo, monkeypatch, force_python_fallback
):
    """``deny = ["Read(secrets/**)"]`` refuses Read. It is a statement about what the
    model may see of those files, so it holds for every other way their lines could
    reach it: matches, the context around them, the names of the files, and the counts."""
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    (tmp_repo / "secrets").mkdir()
    (tmp_repo / "secrets" / "notes.txt").write_text("before\nneedle: launch code 1234\nafter\n")
    (tmp_repo / "open.txt").write_text("needle: public\n")
    denying = _deny_reading(ctx, "Read(secrets/**)")
    for mode in ("content", "files", "count"):
        outcome = await GrepTool().run(
            denying, "t1", {"pattern": "needle", "output_mode": mode, "-C": 1}
        )
        assert not outcome.is_error, outcome.content
        assert "secrets" not in outcome.content, f"{mode}: {outcome.content}"
        assert "launch code" not in outcome.content, f"{mode}: {outcome.content}"
        assert "before" not in outcome.content, f"{mode}: {outcome.content}"
        assert "open.txt" in outcome.content, f"{mode}: {outcome.content}"


async def test_a_deny_rule_for_another_tool_does_not_hide_files_from_grep(ctx, tmp_repo):
    (tmp_repo / "secrets").mkdir()
    (tmp_repo / "secrets" / "notes.txt").write_text("needle\n")
    denying = _deny_reading(ctx, "Write(secrets/**)")
    outcome = await GrepTool().run(denying, "t1", {"pattern": "needle"})
    assert "secrets/notes.txt" in outcome.content


@pytest.mark.parametrize("force_python_fallback", FORCE_PYTHON_FALLBACK)
@pytest.mark.parametrize(
    "path",
    [".npmrc", ".kube/config", ".docker/config.json", "release/app.keystore", "main.tfstate"],
)
async def test_the_credentials_paths_added_to_the_list_are_dropped_from_matches_too(
    ctx, tmp_repo, monkeypatch, force_python_fallback, path
):
    _maybe_force_python_fallback(monkeypatch, force_python_fallback)
    target = tmp_repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("needle: not for the model\n")
    (tmp_repo / "open.txt").write_text("needle: fine\n")
    outcome = await GrepTool().run(ctx, "t1", {"pattern": "needle"})
    assert "not for the model" not in outcome.content
    assert "open.txt" in outcome.content


def test_the_description_is_byte_identical_to_the_spec():
    """It is a prompt. Changing a word needs a design note, so pin the shape."""
    assert GrepTool().spec().description == EXPECTED_DESCRIPTION
