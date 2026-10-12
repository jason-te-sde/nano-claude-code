import os

import pytest

from nanoclaude.tools.base import ToolArgumentError
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


def test_permission_request_raises_on_a_missing_pattern(ctx):
    """A malformed call must raise here, not be tolerated: the dispatch loop's
    first phase is what turns ToolArgumentError into a clean refusal, the same
    way Read's and Edit's require_str-based permission_request already do
    (test_edit.py's test_a_non_string_path_argument_raises_before_any_file_access).
    The registry-wide test (test_registry.py) supplies a valid call instead of
    relying on this method tolerating an incomplete one.
    """
    with pytest.raises(ToolArgumentError, match="pattern must be a string"):
        GlobTool().permission_request(ctx, {})


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


async def test_a_symlink_to_a_file_outside_the_sandbox_is_not_listed(layout):
    (layout.outside / "private.txt").write_text("")
    (layout.project / "notes.txt").symlink_to(layout.outside / "private.txt")
    (layout.project / "real.txt").write_text("")
    outcome = await GlobTool().run(layout.ctx, "t1", {"pattern": "**/*.txt"})
    assert "real.txt" in outcome.content
    assert "notes.txt" not in outcome.content


async def test_nothing_is_listed_through_a_symlink_to_a_directory_outside_the_sandbox(layout):
    (layout.outside / "private.txt").write_text("")
    (layout.project / "vendor").symlink_to(layout.outside, target_is_directory=True)
    (layout.project / "real.txt").write_text("")
    outcome = await GlobTool().run(layout.ctx, "t1", {"pattern": "**/*"})
    assert "real.txt" in outcome.content
    assert "vendor" not in outcome.content
    assert "private.txt" not in outcome.content


async def test_a_symlink_to_a_file_inside_the_sandbox_is_still_listed(layout):
    """The check must not alarm on everything: a link within the project is as visible
    as the file it leads to."""
    (layout.project / "real.txt").write_text("")
    (layout.project / "alias.txt").symlink_to(layout.project / "real.txt")
    outcome = await GlobTool().run(layout.ctx, "t1", {"pattern": "**/*.txt"})
    assert "alias.txt" in outcome.content and "real.txt" in outcome.content


async def test_links_that_lead_outside_do_not_use_up_the_result_limit(layout):
    """Entries are dropped before the limit is applied. Filtering the capped list
    afterwards would let a directory full of fresh links to outside push every real
    file out of the answer."""
    (layout.outside / "private.txt").write_text("")
    for i in range(DEFAULT_LIMIT):
        (layout.project / f"link{i}.txt").symlink_to(layout.outside / "private.txt")
    real = layout.project / "real.txt"
    real.write_text("")
    os.utime(real, (1, 1))  # older than every link
    outcome = await GlobTool().run(layout.ctx, "t1", {"pattern": "*.txt"})
    assert "real.txt" in outcome.content
    assert "link" not in outcome.content


def _count_realpath(monkeypatch: pytest.MonkeyPatch, under: str) -> list[str]:
    """Patch ``os.path.realpath`` to record the paths under ``under`` it is asked about."""
    real = os.path.realpath
    asked: list[str] = []

    def counting(path: str | os.PathLike[str], **kwargs: bool) -> str:
        if str(path).startswith(under) and str(path) != under:
            asked.append(str(path))
        return real(path, **kwargs)

    monkeypatch.setattr(os.path, "realpath", counting)
    return asked


async def test_a_tree_with_no_links_costs_no_realpath_at_all(layout, monkeypatch):
    """One ``realpath`` is a system call for every component of the path, and the walk does
    not enter a directory link, so an entry that is not a link cannot lead outside."""
    for directory in ("a", "a/b", "c"):
        (layout.project / directory).mkdir(parents=True, exist_ok=True)
        for i in range(5):
            (layout.project / directory / f"f{i}.py").write_text("")
    asked = _count_realpath(monkeypatch, str(layout.project))
    outcome = await GlobTool().run(layout.ctx, "t1", {"pattern": "**/*.py"})
    assert "15 file(s)" in outcome.content, outcome.content
    assert asked == []


async def test_a_tree_with_one_link_costs_one_realpath_and_the_link_is_judged(layout, monkeypatch):
    for i in range(10):
        (layout.project / f"f{i}.py").write_text("")
    (layout.outside / "private.py").write_text("")
    (layout.project / "escape.py").symlink_to(layout.outside / "private.py")
    asked = _count_realpath(monkeypatch, str(layout.project))
    outcome = await GlobTool().run(layout.ctx, "t1", {"pattern": "*.py"})
    assert asked == [str(layout.project / "escape.py")]
    assert "10 file(s)" in outcome.content and "escape.py" not in outcome.content


async def test_a_link_to_a_file_inside_the_root_costs_one_realpath_and_is_listed(
    layout, monkeypatch
):
    (layout.project / "real.py").write_text("")
    (layout.project / "alias.py").symlink_to(layout.project / "real.py")
    asked = _count_realpath(monkeypatch, str(layout.project))
    outcome = await GlobTool().run(layout.ctx, "t1", {"pattern": "*.py"})
    assert asked == [str(layout.project / "alias.py")]
    assert "alias.py" in outcome.content and "real.py" in outcome.content


def test_the_description_is_byte_identical_to_the_spec():
    """It is a prompt. Changing a word needs a design note, so pin the shape."""
    assert GlobTool().spec().description == EXPECTED_DESCRIPTION
