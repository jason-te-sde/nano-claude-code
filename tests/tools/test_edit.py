from dataclasses import replace as dc_replace

import pytest

from nanoclaude.tools.base import ToolArgumentError
from nanoclaude.tools.edit import EditTool
from nanoclaude.tools.fs import stamp_of


def having_read(ctx, path):
    return dc_replace(ctx, read_state={str(path): stamp_of(str(path))})


async def test_editing_without_reading_first_is_refused(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\n")
    outcome = await EditTool().run(
        ctx, "t1", {"path": "a.py", "edits": [{"old_string": "1", "new_string": "2"}]}
    )
    assert outcome.is_error and "has not been read" in outcome.content
    assert (tmp_repo / "a.py").read_text() == "x = 1\n"


async def test_a_successful_edit_writes_and_returns_a_diff(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {"path": "a.py", "edits": [{"old_string": "x = 1", "new_string": "x = 2"}]},
    )
    assert not outcome.is_error
    assert target.read_text() == "x = 2\n"
    assert "-x = 1" in outcome.content and "+x = 2" in outcome.content


async def test_a_failing_edit_in_a_batch_leaves_the_file_byte_identical(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    original = "a = 1\nb = 2\n"
    target.write_text(original)
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {
            "path": "a.py",
            "edits": [
                {"old_string": "a = 1", "new_string": "a = 9"},
                {"old_string": "nonexistent", "new_string": "x"},
            ],
        },
    )
    assert outcome.is_error and "was not found" in outcome.content
    assert target.read_text() == original


async def test_editing_a_file_that_changed_on_disk_is_refused(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("v1\n")
    ctx = having_read(ctx, target)
    target.write_text("v2\n")
    outcome = await EditTool().run(
        ctx, "t1", {"path": "a.py", "edits": [{"old_string": "v1", "new_string": "v3"}]}
    )
    assert outcome.is_error and "changed since" in outcome.content


async def test_the_new_stamp_is_reported_so_a_second_edit_can_follow(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {"path": "a.py", "edits": [{"old_string": "1", "new_string": "2"}]},
    )
    ((_path, stamp),) = outcome.observed
    assert stamp == stamp_of(str(target))


async def test_malformed_edits_argument_is_a_clear_error(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x\n")
    outcome = await EditTool().run(
        having_read(ctx, target), "t1", {"path": "a.py", "edits": "not a list"}
    )
    assert outcome.is_error and "edits must be a list" in outcome.content


# The tests below are not in the brief. They close gaps: branches the brief's
# seven given tests never reach, each guarded by an `or` whose two halves need
# separate tests because line coverage cannot see a dropped half, plus the
# tool's own refusal and permission-request behavior that "Produces: WriteTool,
# EditTool" calls for but no given test exercises.


async def test_an_empty_edits_list_is_a_clear_error(ctx, tmp_repo):
    """`not isinstance(raw, list) or not raw` -- the given malformed-argument
    test only ever makes the first half true (a string, not a list at all).
    An empty list makes isinstance true and only the second half catch it."""
    target = tmp_repo / "a.py"
    target.write_text("x\n")
    outcome = await EditTool().run(having_read(ctx, target), "t1", {"path": "a.py", "edits": []})
    assert outcome.is_error and "edits must be a list" in outcome.content


async def test_a_non_object_edit_entry_is_a_clear_error(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x\n")
    outcome = await EditTool().run(
        having_read(ctx, target), "t1", {"path": "a.py", "edits": ["not an object"]}
    )
    assert outcome.is_error and "edit 1 must be an object" in outcome.content


async def test_a_non_string_old_string_is_a_clear_error(ctx, tmp_repo):
    """`not isinstance(old, str) or not isinstance(new, str)` -- this makes
    only the first half true."""
    target = tmp_repo / "a.py"
    target.write_text("x\n")
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {"path": "a.py", "edits": [{"old_string": 1, "new_string": "y"}]},
    )
    assert outcome.is_error and "old_string and new_string must be strings" in outcome.content


async def test_a_non_string_new_string_is_a_clear_error(ctx, tmp_repo):
    """The other half of the same `or`: old_string is fine, new_string is not."""
    target = tmp_repo / "a.py"
    target.write_text("x\n")
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {"path": "a.py", "edits": [{"old_string": "x", "new_string": 2}]},
    )
    assert outcome.is_error and "old_string and new_string must be strings" in outcome.content


def test_a_non_string_path_argument_raises_before_any_file_access(ctx):
    """_require_path's own guard: a malformed path is a programming error the
    dispatch loop is expected to catch, the same way ReadTool and WriteTool
    let require_str's ToolArgumentError propagate uncaught from
    permission_request (tools/base.py; tools/read.py)."""
    with pytest.raises(ToolArgumentError, match="path must be a string, got int"):
        EditTool().permission_request(ctx, {"path": 123})


async def test_editing_a_file_that_was_deleted_since_it_was_read_says_so(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    ctx = having_read(ctx, target)
    target.unlink()
    outcome = await EditTool().run(
        ctx, "t1", {"path": "a.py", "edits": [{"old_string": "x = 1", "new_string": "x = 2"}]}
    )
    assert outcome.is_error and "no longer exists" in outcome.content


async def test_editing_a_file_too_large_to_read_is_refused_through_the_tool(ctx, tmp_repo):
    target = tmp_repo / "big.txt"
    target.write_bytes(b"x" * 1_000_001)
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {"path": "big.txt", "edits": [{"old_string": "x", "new_string": "y"}]},
    )
    assert outcome.is_error and "Grep" in outcome.content


async def test_an_edit_that_produces_no_change_is_refused(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {"path": "a.py", "edits": [{"old_string": "x = 1", "new_string": "x = 1"}]},
    )
    assert outcome.is_error and "produced no change" in outcome.content
    assert target.read_text() == "x = 1\n"


async def test_replace_all_is_honored_through_the_tool(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x\nx\nx\n")
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {
            "path": "a.py",
            "edits": [{"old_string": "x", "new_string": "y", "replace_all": True}],
        },
    )
    assert not outcome.is_error
    assert target.read_text() == "y\ny\ny\n"


def test_the_permission_request_marks_this_as_a_write(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    request = EditTool().permission_request(
        ctx, {"path": "a.py", "edits": [{"old_string": "x", "new_string": "y"}]}
    )
    assert request.resolved_paths == (str(tmp_repo / "a.py"),)
    assert request.is_write is True


def test_the_description_is_the_one_from_the_brief():
    description = EditTool().spec().description
    assert description.startswith("Make one or more exact string replacements in a single file.")
    assert "Never include the line-number prefix" in description
    assert "Preserve the file's existing indentation and line endings" in description
