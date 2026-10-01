from nanoclaude.tools.glob import GlobTool
from nanoclaude.tools.search import DEFAULT_LIMIT

EXPECTED_DESCRIPTION = """Find files by glob pattern, most recently modified first.

- `pattern` is a glob such as `src/**/*.py` or `**/test_*.py`.
- `path` limits the search to a subdirectory (default: the working directory).
- Paths ignored by .gitignore are skipped. At most 200 results are returned."""


async def test_matching_files_are_listed(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("")
    (tmp_repo / "b.txt").write_text("")
    outcome = await GlobTool().run(ctx, "t1", {"pattern": "**/*.py"})
    assert "a.py" in outcome.content and "b.txt" not in outcome.content


async def test_no_matches_says_so_rather_than_returning_nothing(ctx):
    outcome = await GlobTool().run(ctx, "t1", {"pattern": "**/*.rs"})
    assert not outcome.is_error and "no files matched" in outcome.content.lower()


def test_glob_is_read_only_and_needs_no_paths_resolved(ctx):
    request = GlobTool().permission_request(ctx, {"pattern": "**/*.py"})
    assert GlobTool().read_only and request.is_write is False


async def test_a_path_argument_limits_the_search_to_a_subdirectory(ctx, tmp_repo):
    (tmp_repo / "sub").mkdir()
    (tmp_repo / "sub" / "a.py").write_text("")
    (tmp_repo / "top.py").write_text("")
    outcome = await GlobTool().run(ctx, "t1", {"pattern": "*.py", "path": "sub"})
    assert "a.py" in outcome.content
    assert "top.py" not in outcome.content


def test_the_permission_request_resolves_a_path_argument_when_given(ctx, tmp_repo):
    request = GlobTool().permission_request(ctx, {"pattern": "**/*.py", "path": "sub"})
    assert request.resolved_paths == (ctx.resolve("sub"),)


async def test_hitting_the_result_limit_says_so_in_the_header(ctx, tmp_repo):
    for i in range(DEFAULT_LIMIT):
        (tmp_repo / f"f{i}.py").write_text("")
    outcome = await GlobTool().run(ctx, "t1", {"pattern": "*.py"})
    assert "truncated at 200" in outcome.content


def test_the_description_is_byte_identical_to_the_spec():
    """It is a prompt. Changing a word needs a design note, so pin the shape."""
    assert GlobTool().spec().description == EXPECTED_DESCRIPTION
