import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from nanoclaude.permissions.redact import PRIVATE_KEY_MARKER, Redactor
from nanoclaude.tools.read import ReadTool
from tests.fifo import call_without_blocking, make_fifo
from tests.keys import pem


async def test_output_is_line_numbered_with_a_single_tab(ctx, tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\ny = 2\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "a.py"})
    assert not outcome.is_error
    assert "1\tx = 1" in outcome.content
    assert "2\ty = 2" in outcome.content


async def test_a_line_that_assigns_the_result_of_a_call_to_a_secret_name_is_shown_as_it_is(
    ctx, tmp_repo
):
    """Edit matches on content: a line the model read with its call redacted is a line it
    can no longer edit."""
    line = "password = get_password_from_env()"
    (tmp_repo / "settings.py").write_text(f"{line}\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "settings.py"})
    assert f"1\t{line}" in outcome.content


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


# ---- a window of a file never shows part of a private key block


def _file_with_a_key(root: Path, body: int = 6) -> tuple[list[str], list[str]]:
    block, rows = pem(lines=body)
    lines = ["head 1", "head 2", *block, "tail 1", "tail 2"]
    (root / "deploy.txt").write_text("\n".join(lines) + "\n")
    return lines, rows


@pytest.mark.parametrize("offset", [1, 2, 3, 4, 5, 8, 9, 10, 11, 12])
@pytest.mark.parametrize("limit", [1, 2, 3, 5, 100])
async def test_no_window_of_a_file_shows_a_line_of_the_key_in_it(ctx, tmp_repo, offset, limit):
    _, rows = _file_with_a_key(tmp_repo)
    outcome = await ReadTool().run(
        ctx, "t1", {"path": "deploy.txt", "offset": offset, "limit": limit}
    )
    assert not outcome.is_error, outcome.content
    assert not any(row in outcome.content for row in rows), outcome.content
    assert "BEGIN" not in outcome.content and "END RSA" not in outcome.content, outcome.content


async def test_a_window_that_starts_past_the_begin_line_is_one_mask(ctx, tmp_repo):
    _file_with_a_key(tmp_repo)
    outcome = await ReadTool().run(ctx, "t1", {"path": "deploy.txt", "offset": 6, "limit": 4})
    body = outcome.content.split("\n", 1)[1]
    assert body == f"6\t{PRIVATE_KEY_MARKER}"


async def test_a_whole_file_shows_the_key_as_one_mask_and_every_other_line(ctx, tmp_repo):
    lines, _ = _file_with_a_key(tmp_repo)
    outcome = await ReadTool().run(ctx, "t1", {"path": "deploy.txt"})
    body = outcome.content.split("\n", 1)[1].split("\n")
    assert body == [
        "1\thead 1",
        "2\thead 2",
        f"3\t{PRIVATE_KEY_MARKER}",
        f"{len(lines) - 1}\ttail 1",
        f"{len(lines)}\ttail 2",
    ]


async def test_the_header_still_says_which_lines_the_window_covered(ctx, tmp_repo):
    lines, _ = _file_with_a_key(tmp_repo)
    outcome = await ReadTool().run(ctx, "t1", {"path": "deploy.txt", "offset": 5, "limit": 3})
    assert f"({len(lines)} lines), showing 5-7" in outcome.content


async def test_the_stamp_covers_the_whole_file_whatever_the_window_masked(ctx, tmp_repo):
    _file_with_a_key(tmp_repo)
    outcome = await ReadTool().run(ctx, "t1", {"path": "deploy.txt", "offset": 6, "limit": 2})
    ((_, stamp),) = outcome.observed
    assert stamp.size == (tmp_repo / "deploy.txt").stat().st_size


async def test_a_file_with_no_key_in_it_is_shown_exactly_as_before(ctx, tmp_repo):
    (tmp_repo / "a.txt").write_text("one\ntwo\nthree\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "a.txt", "offset": 2, "limit": 1})
    assert outcome.content.endswith("\n2\ttwo")


async def test_with_redaction_switched_off_a_window_shows_what_is_there(ctx, tmp_repo):
    _, rows = _file_with_a_key(tmp_repo)
    plain = replace(ctx, redactor=Redactor(enabled=False))
    outcome = await ReadTool().run(plain, "t1", {"path": "deploy.txt", "offset": 6, "limit": 2})
    assert rows[2] in outcome.content


async def test_a_key_in_a_file_with_windows_line_endings_is_masked_in_a_window(ctx, tmp_repo):
    block, rows = pem()
    (tmp_repo / "win.txt").write_bytes("\r\n".join(["a", *block, "b"]).encode() + b"\r\n")
    outcome = await ReadTool().run(ctx, "t1", {"path": "win.txt", "offset": 4, "limit": 2})
    assert not any(row in outcome.content for row in rows)


async def test_a_line_before_the_key_that_ends_with_text_the_model_must_edit_stays(ctx, tmp_repo):
    block, rows = pem(lines=2)
    (tmp_repo / "cfg.py").write_text(
        f'KEY = """{block[0]}\n' + "\n".join(block[1:]) + '"""\nNEXT = 1\n'
    )
    outcome = await ReadTool().run(ctx, "t1", {"path": "cfg.py"})
    assert f'1\tKEY = """{PRIVATE_KEY_MARKER}' in outcome.content
    assert "NEXT = 1" in outcome.content and not any(row in outcome.content for row in rows)
