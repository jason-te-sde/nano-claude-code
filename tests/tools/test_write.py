from nanoclaude.tools.fs import stamp_of
from nanoclaude.tools.write import WriteTool


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
