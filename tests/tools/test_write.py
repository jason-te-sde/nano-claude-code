import asyncio

import pytest

from nanoclaude.tools.base import ToolArgumentError
from nanoclaude.tools.fs import stamp_of
from nanoclaude.tools.write import WriteTool, preview_write
from tests.fifo import call_without_blocking, make_fifo


async def test_creating_a_new_file_needs_no_prior_read(ctx, tmp_repo):
    outcome = await WriteTool().run(ctx, "t1", {"path": "new.py", "content": "x = 1\n"})
    assert not outcome.is_error
    assert (tmp_repo / "new.py").read_text() == "x = 1\n"
    assert "Created" in outcome.content


async def test_overwriting_an_unread_file_is_refused(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("important\n")
    outcome = await WriteTool().run(ctx, "t1", {"path": "a.py", "content": "gone\n"})
    assert outcome.is_error and "has not been read" in outcome.content
    assert (tmp_repo / "a.py").read_text() == "important\n"


async def test_overwriting_a_file_that_changed_since_it_was_read_is_refused(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("v1\n")
    stale = stamp_of(str(target))
    target.write_text("v2 written by someone else\n")
    seen = {str(target): stale}
    from dataclasses import replace as dc_replace

    outcome = await WriteTool().run(
        dc_replace(ctx, read_state=seen), "t1", {"path": "a.py", "content": "v3\n"}
    )
    assert outcome.is_error and "changed since" in outcome.content
    assert target.read_text() == "v2 written by someone else\n"


async def test_a_successful_write_reports_the_new_stamp(ctx, tmp_repo):
    outcome = await WriteTool().run(ctx, "t1", {"path": "new.py", "content": "x = 1\n"})
    ((path, stamp),) = outcome.observed
    assert path == str(tmp_repo / "new.py") and stamp.size == 6


# The tests below are not in the brief. The brief's four given tests only ever
# exercise the "file did not exist" (Created) half of
# `verb = "Updated" if current is not None else "Created"`, and only the
# trailing-newline half of `content.endswith("\n") or not content` -- a
# dropped "Updated" or a dropped half of that `or` would still pass every
# given test, and line coverage alone cannot see it since each is one line.


async def test_overwriting_a_file_that_was_read_and_is_unchanged_succeeds(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("v1\n")
    from dataclasses import replace as dc_replace

    seen = {str(target): stamp_of(str(target))}
    outcome = await WriteTool().run(
        dc_replace(ctx, read_state=seen), "t1", {"path": "a.py", "content": "v2\n"}
    )
    assert not outcome.is_error
    assert "Updated" in outcome.content
    assert target.read_text() == "v2\n"


async def test_the_line_count_excludes_a_trailing_newline(ctx, tmp_repo):
    outcome = await WriteTool().run(ctx, "t1", {"path": "a.py", "content": "a\nb\n"})
    assert "2 lines" in outcome.content


async def test_an_empty_file_is_reported_as_zero_lines(ctx, tmp_repo):
    outcome = await WriteTool().run(ctx, "t1", {"path": "empty.py", "content": ""})
    assert "0 lines" in outcome.content


async def test_content_without_a_trailing_newline_still_counts_as_a_line(ctx, tmp_repo):
    outcome = await WriteTool().run(ctx, "t1", {"path": "a.py", "content": "a"})
    assert "1 lines" in outcome.content


def test_the_permission_request_marks_this_as_a_write(ctx, tmp_repo):
    request = WriteTool().permission_request(ctx, {"path": "a.py", "content": "x\n"})
    assert request.resolved_paths == (str(tmp_repo / "a.py"),)
    assert request.is_write is True


def test_the_description_is_the_one_from_the_brief():
    """It is a prompt. Changing a word needs a design note, so pin the shape."""
    description = WriteTool().spec().description
    assert description.startswith("Write a UTF-8 text file, creating it or replacing it entirely.")
    assert "Prefer Edit for changing part of a file" in description
    assert "The write is atomic" in description


# --------------------------------------------------------------------------
# Previewing: what a confirmation shows before anything is written
# --------------------------------------------------------------------------


def test_a_preview_of_a_new_file_says_what_will_be_created_and_writes_nothing(tmp_repo):
    target = tmp_repo / "new.py"
    preview = preview_write(str(target), "new.py", {"path": "new.py", "content": "a = 1\nb = 2\n"})
    assert (preview.kind, preview.lines, preview.size) == ("create", 2, 12)
    assert preview.content == "a = 1\nb = 2\n" and preview.diff == ""
    assert not target.exists()


def test_a_preview_counts_bytes_and_not_characters(tmp_repo):
    content = "caf\N{LATIN SMALL LETTER E WITH ACUTE}\n"
    preview = preview_write(str(tmp_repo / "n.txt"), "n.txt", {"path": "n.txt", "content": content})
    assert (preview.lines, preview.size) == (1, 6)


def test_a_preview_of_a_replacement_is_the_diff_against_what_is_there(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\ny = 2\n")
    preview = preview_write(str(target), "a.py", {"path": "a.py", "content": "x = 1\ny = 3\n"})
    assert preview.kind == "replace"
    assert "--- a/a.py" in preview.diff and "-y = 2" in preview.diff and "+y = 3" in preview.diff
    assert target.read_text() == "x = 1\ny = 2\n"


def test_a_preview_of_the_same_content_says_nothing_will_change(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    preview = preview_write(str(target), "a.py", {"path": "a.py", "content": "x = 1\n"})
    assert preview.kind == "unchanged" and preview.diff == ""


def test_a_preview_over_a_file_that_cannot_be_compared_says_why(tmp_repo):
    binary = tmp_repo / "blob.bin"
    binary.write_bytes(b"\0\1\2")
    preview = preview_write(str(binary), "blob.bin", {"path": "blob.bin", "content": "text\n"})
    assert preview.kind == "overwrite" and "looks binary" in preview.why
    assert (preview.lines, preview.size) == (1, 5)
    folder = tmp_repo / "folder"
    folder.mkdir()
    preview = preview_write(str(folder), "folder", {"path": "folder", "content": "text\n"})
    assert preview.kind == "overwrite" and "Is a directory" in preview.why


@pytest.mark.parametrize(
    "arguments",
    [{"path": "a.py"}, {"path": "a.py", "content": 3}, {"path": "a.py", "content": "\ud800"}],
    ids=["no content", "content not text", "content that is not valid text"],
)
def test_a_preview_of_a_call_that_cannot_be_written_is_an_argument_error(tmp_repo, arguments):
    with pytest.raises(ToolArgumentError):
        preview_write(str(tmp_repo / "a.py"), "a.py", arguments)


@pytest.mark.parametrize("content", ["", "a", "a\n", "a\nb", "a\nb\n", "\n", "a\n\n"])
async def test_the_preview_counts_lines_the_way_the_tool_reports_them(ctx, tmp_repo, content):
    preview = preview_write(str(tmp_repo / "n.txt"), "n.txt", {"path": "n.txt", "content": content})
    outcome = await WriteTool().run(ctx, "t1", {"path": "n.txt", "content": content})
    assert f"({preview.size} bytes, {preview.lines} lines)" in outcome.content


def test_a_preview_over_a_fifo_says_why_it_cannot_be_compared_and_does_not_wait(tmp_repo):
    fifo = make_fifo(tmp_repo / "pipe.fifo")
    preview = call_without_blocking(
        fifo, preview_write, str(fifo), "pipe.fifo", {"path": "pipe.fifo", "content": "x\n"}
    )
    assert preview.kind == "overwrite" and preview.why == f"{fifo} is not a regular file"


def test_writing_over_a_fifo_is_an_error_result_and_does_not_wait_for_a_reader(ctx, tmp_repo):
    fifo = make_fifo(tmp_repo / "pipe.fifo")
    outcome = call_without_blocking(
        fifo, asyncio.run, WriteTool().run(ctx, "t1", {"path": "pipe.fifo", "content": "x\n"})
    )
    assert outcome.is_error and "pipe.fifo is not a regular file" in outcome.content
