from dataclasses import replace as dc_replace

import pytest

from nanoclaude.tools.base import ToolArgumentError
from nanoclaude.tools.edit import EditError, EditTool, preview_edit, unified_diff
from nanoclaude.tools.fs import BinaryFileError, stamp_of


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


# --------------------------------------------------------------------------
# Previewing: the diff a confirmation shows before anything is written
# --------------------------------------------------------------------------


def test_a_preview_shows_the_change_and_writes_nothing(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\ny = 2\n")
    diff = preview_edit(
        str(target), "a.py", {"edits": [{"old_string": "x = 1", "new_string": "x = 9"}]}
    )
    assert "--- a/a.py" in diff and "-x = 1" in diff and "+x = 9" in diff
    assert target.read_text() == "x = 1\ny = 2\n"


async def test_the_preview_is_the_diff_the_edit_then_makes(ctx, tmp_repo):
    # Two code paths, one answer: what a person approves is what gets written, so the
    # preview cannot be allowed to drift from the tool.
    target = tmp_repo / "a.py"
    target.write_text("def f():\n    return 1\n\n\ndef g():\n    return 1\n")
    arguments = {
        "path": "a.py",
        "edits": [
            {"old_string": "def f():\n    return 1", "new_string": "def f():\n    return 2"},
            {"old_string": "g", "new_string": "h"},
        ],
    }
    preview = preview_edit(str(target), "a.py", arguments)
    outcome = await EditTool().run(having_read(ctx, target), "t1", arguments)
    assert not outcome.is_error
    assert outcome.content == f"Edited a.py\n{preview}"


def test_a_preview_of_an_edit_that_cannot_apply_says_why_as_the_edit_would(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x\nx\n")
    with pytest.raises(EditError, match="appears 2 times"):
        preview_edit(str(target), "a.py", {"edits": [{"old_string": "x", "new_string": "y"}]})
    with pytest.raises(EditError, match="was not found"):
        preview_edit(str(target), "a.py", {"edits": [{"old_string": "z", "new_string": "y"}]})


def test_a_preview_of_an_edit_that_changes_nothing_is_refused_like_the_edit(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x\n")
    with pytest.raises(EditError, match="produced no change"):
        preview_edit(str(target), "a.py", {"edits": [{"old_string": "x", "new_string": "x"}]})


def test_a_preview_of_malformed_arguments_is_an_argument_error(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x\n")
    with pytest.raises(ToolArgumentError, match="edits must be a list"):
        preview_edit(str(target), "a.py", {"edits": "x"})


def test_a_preview_of_a_file_that_cannot_be_read_is_an_os_error(tmp_repo):
    edits = {"edits": [{"old_string": "x", "new_string": "y"}]}
    with pytest.raises(FileNotFoundError):
        preview_edit(str(tmp_repo / "missing.py"), "missing.py", edits)
    binary = tmp_repo / "blob.bin"
    binary.write_bytes(b"\0\1\2")
    with pytest.raises(BinaryFileError):
        preview_edit(str(binary), "blob.bin", edits)


# --------------------------------------------------------------------------
# The diff of a file whose last line has no newline
# --------------------------------------------------------------------------


def test_a_last_line_without_a_newline_is_marked_and_not_run_into_the_next_line():
    # difflib leaves the line as it is, so "-x = 1" and "+x = 2" came out as one line:
    # "-x = 1+x = 2", which a person reads as one removed line.
    assert unified_diff("x = 1", "x = 2", "a.py") == (
        "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n"
        "-x = 1\n\\ No newline at end of file\n"
        "+x = 2\n\\ No newline at end of file\n"
    )


def test_adding_a_final_newline_is_a_change_the_diff_shows():
    assert unified_diff("a\nb", "a\nb\n", "f") == (
        "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n a\n-b\n\\ No newline at end of file\n+b\n"
    )


def test_removing_a_final_newline_is_a_change_the_diff_shows():
    assert unified_diff("a\nb\n", "a\nb", "f") == (
        "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n a\n-b\n+b\n\\ No newline at end of file\n"
    )


def test_only_a_newline_ends_a_line_in_a_diff():
    # A lone carriage return or a form feed is a character in a line, not the end of one.
    diff = unified_diff("a\rb\n\x0c\n", "a\rc\n\x0c\n", "f")
    assert diff == "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n-a\rb\n+a\rc\n \x0c\n"


def test_an_empty_file_has_no_lines_to_diff():
    assert unified_diff("", "", "f") == ""
    assert unified_diff("", "new\n", "f") == "--- a/f\n+++ b/f\n@@ -0,0 +1 @@\n+new\n"


async def test_the_diff_an_edit_reports_marks_a_last_line_without_a_newline(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1")
    outcome = await EditTool().run(
        having_read(ctx, target),
        "t1",
        {"path": "a.py", "edits": [{"old_string": "1", "new_string": "2"}]},
    )
    assert (
        not outcome.is_error and "-x = 1\n\\ No newline at end of file\n+x = 2\n" in outcome.content
    )
