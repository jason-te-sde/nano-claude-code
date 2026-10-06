import asyncio

from nanoclaude.tools.read import ReadTool
from tests.fifo import call_without_blocking, make_fifo


async def test_output_is_line_numbered_with_a_single_tab(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\ny = 2\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "a.py"})
    assert not outcome.is_error
    assert "1\tx = 1" in outcome.content
    assert "2\ty = 2" in outcome.content


async def test_a_window_reports_which_lines_it_showed(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("\n".join(f"line {i}" for i in range(1, 101)) + "\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "a.py", "offset": 50, "limit": 2})
    assert "50\tline 50" in outcome.content and "51\tline 51" in outcome.content
    assert "52\t" not in outcome.content
    assert "100 lines" in outcome.content


async def test_reading_records_a_stamp_for_the_whole_file_not_the_window(ctx, tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\ny = 2\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "a.py", "limit": 1})
    ((path, stamp),) = outcome.observed
    assert path == str(target)
    assert stamp.size == len("x = 1\ny = 2\n")


async def test_a_missing_file_is_an_error_result_not_an_exception(ctx):
    outcome = await ReadTool().run(ctx, "t1", {"path": "nope.py"})
    assert outcome.is_error and "does not exist" in outcome.content


async def test_reading_a_directory_points_at_glob_instead(ctx, tmp_repo):
    (tmp_repo / "sub").mkdir()
    outcome = await ReadTool().run(ctx, "t1", {"path": "sub"})
    assert outcome.is_error and "is a directory" in outcome.content


async def test_a_file_too_large_to_read_is_refused_through_the_tool(ctx, tmp_repo):
    """fs.read_text's FileSystemError subclasses are covered directly in
    test_fs.py; this pins that run() actually passes the message through
    (the `except FileSystemError as exc: return failed(call_id, str(exc))`
    branch) instead of letting it escape as an exception.
    """
    (tmp_repo / "big.txt").write_bytes(b"x" * 1_000_001)
    outcome = await ReadTool().run(ctx, "t1", {"path": "big.txt"})
    assert outcome.is_error and "Grep" in outcome.content


async def test_an_offset_past_the_end_says_how_long_the_file_is(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("one\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "a.py", "offset": 99})
    assert outcome.is_error and "1 lines" in outcome.content


async def test_an_empty_file_reads_as_empty_rather_than_failing(ctx, tmp_repo):
    (tmp_repo / "empty.py").write_text("")
    outcome = await ReadTool().run(ctx, "t1", {"path": "empty.py"})
    assert not outcome.is_error and "(empty file)" in outcome.content


async def test_very_long_lines_are_clipped_with_a_count(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("x = '" + "a" * 3000 + "'\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "a.py"})
    assert "more characters]" in outcome.content


def test_the_permission_request_carries_the_resolved_path(ctx, tmp_repo):
    request = ReadTool().permission_request(ctx, {"path": "a.py"})
    assert request.resolved_paths == (str(tmp_repo / "a.py"),)
    assert request.is_write is False


def test_the_description_is_the_one_from_the_spec():
    """It is a prompt. Changing a word needs a design note, so pin the shape."""
    description = ReadTool().spec().description
    assert description.startswith("Read a UTF-8 text file from the working directory.")
    assert "line number and a single tab" in description
    assert "Reading a file records what it contained" in description


def test_reading_a_fifo_is_an_error_result_and_does_not_wait_for_a_writer(ctx, tmp_repo):
    # Not a coroutine: if the tool did open it, the event loop would be what blocked.
    fifo = make_fifo(tmp_repo / "pipe.fifo")
    outcome = call_without_blocking(
        fifo, asyncio.run, ReadTool().run(ctx, "t1", {"path": "pipe.fifo"})
    )
    assert outcome.is_error and "pipe.fifo is not a regular file" in outcome.content
