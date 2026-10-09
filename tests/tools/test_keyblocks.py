"""Which lines of a file are inside a private key block, and what is shown in their place.

A window of a file (what Read shows with an offset and a limit, what Grep shows of a match and
its context) used to be scrubbed on its own, so a window that began or ended inside a block
showed the body of the key: nothing in it was recognisable as one. The extent of a block is
now worked out on the whole file, and the lines of a window that are in one are masked.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterator

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nanoclaude.permissions.redact import PRIVATE_KEY_MARKER, Redactor
from nanoclaude.tools.keyblocks import _scan, mask_window, masked_line_numbers
from tests.keys import pem

REDACTOR = Redactor()


def window(lines: list[str], offset: int = 1, limit: int = 2000) -> list[tuple[int, str]]:
    content = "\n".join(lines) + "\n"
    return mask_window(REDACTOR, content, content.splitlines(), offset, limit)


def file_with_a_key(before: int = 2, after: int = 2, body: int = 6) -> tuple[list[str], list[str]]:
    """``before`` lines of text, a key whose body has ``body`` lines, then ``after`` lines."""
    block, rows = pem(lines=body)
    lines = (
        [f"before {i}" for i in range(1, before + 1)]
        + block
        + [f"after {i}" for i in range(1, after + 1)]
    )
    return lines, rows


# ---- Read's window


def test_a_file_with_no_key_is_shown_as_it_is():
    lines = [f"line {i}" for i in range(1, 8)]
    assert window(lines) == list(enumerate(lines, 1))
    assert window(lines, 3, 2) == [(3, "line 3"), (4, "line 4")]


def test_a_whole_key_in_the_window_is_one_masked_line_and_the_numbers_go_on_after_it():
    lines, rows = file_with_a_key()
    shown = window(lines)
    assert shown == [
        (1, "before 1"),
        (2, "before 2"),
        (3, PRIVATE_KEY_MARKER),
        (11, "after 1"),
        (12, "after 2"),
    ]
    assert not any(row in text for _, text in shown for row in rows)


@pytest.mark.parametrize("offset", [4, 5, 7, 9])
def test_a_window_that_starts_inside_the_block_is_masked_over_the_part_it_shows(offset):
    lines, rows = file_with_a_key()
    shown = window(lines, offset)
    assert not any(row in text for _, text in shown for row in rows)
    assert shown[0] == (offset, PRIVATE_KEY_MARKER)
    assert shown[-2:] == [(11, "after 1"), (12, "after 2")]


def test_a_window_that_starts_on_the_end_line_shows_one_mask_and_then_the_file():
    lines, _ = file_with_a_key()
    assert window(lines, 10)[0] == (10, PRIVATE_KEY_MARKER)


@pytest.mark.parametrize("limit", [3, 4, 5, 8])
def test_a_window_that_ends_inside_the_block_is_masked_over_the_part_it_shows(limit):
    lines, rows = file_with_a_key()
    shown = window(lines, 1, limit)
    assert not any(row in text for _, text in shown for row in rows)
    assert not any("BEGIN" in text for _, text in shown)
    assert shown[-1] == (3, PRIVATE_KEY_MARKER)


def test_a_window_of_nothing_but_body_lines_is_one_mask():
    lines, _ = file_with_a_key()
    assert window(lines, 5, 3) == [(5, PRIVATE_KEY_MARKER)]


def test_two_keys_one_straight_after_the_other_are_two_masks():
    first, _ = pem(lines=2)
    second, _ = pem("EC PRIVATE KEY", lines=2)
    assert window([*first, *second, "after"]) == [
        (1, PRIVATE_KEY_MARKER),
        (5, PRIVATE_KEY_MARKER),
        (9, "after"),
    ]


def test_text_on_the_lines_that_open_and_close_a_block_stays_and_the_key_does_not():
    block, rows = pem(lines=3)
    lines = [f'key = """{block[0]}', *block[1:-1], f'{block[-1]}""" # end', "next = 1"]
    shown = window(lines)
    assert shown == [(1, f'key = """{PRIVATE_KEY_MARKER}'), (5, '""" # end'), (6, "next = 1")]
    shown_from_the_middle = window(lines, 3)
    assert shown_from_the_middle[0] == (3, PRIVATE_KEY_MARKER)
    assert not any(row in text for _, text in shown_from_the_middle for row in rows)


def test_two_keys_are_two_masks_and_what_is_between_them_stays():
    first, _ = pem(lines=2)
    second, _ = pem("EC PRIVATE KEY", lines=3)
    lines = [*first, "between", *second, "after"]
    assert window(lines) == [
        (1, PRIVATE_KEY_MARKER),
        (5, "between"),
        (6, PRIVATE_KEY_MARKER),
        (11, "after"),
    ]


def test_a_key_cut_off_before_its_end_is_masked_as_far_as_it_goes():
    block, _ = pem(lines=4)
    lines = ["head", *block[:-1]]  # no END line: the file stops
    shown = window(lines)
    assert shown == [(1, "head"), (2, PRIVATE_KEY_MARKER)]
    assert window(lines, 4) == [(4, PRIVATE_KEY_MARKER)]


def test_a_key_in_a_file_with_windows_line_endings_is_masked_in_a_window():
    block, rows = pem()
    content = "\r\n".join(["a", *block, "b"]) + "\r\n"
    shown = mask_window(REDACTOR, content, content.splitlines(), 4, 2)
    assert not any(row in text for _, text in shown for row in rows)


def test_a_key_in_a_json_string_on_one_line_is_masked_and_the_rest_of_the_line_stays():
    block, _ = pem(lines=2)
    line = '{"key": "' + "\\n".join(block) + '", "id": 7}'
    assert window(["x", line, "y"]) == [
        (1, "x"),
        (2, f'{{"key": "{PRIVATE_KEY_MARKER}", "id": 7}}'),
        (3, "y"),
    ]


def test_nothing_is_masked_when_redaction_is_off():
    lines, _ = file_with_a_key()
    content = "\n".join(lines) + "\n"
    shown = mask_window(Redactor(enabled=False), content, content.splitlines(), 5, 3)
    assert shown == [(5, lines[4]), (6, lines[5]), (7, lines[6])]


def test_an_empty_line_inside_a_block_is_inside_it():
    """The blank line an encrypted key has after its headers."""
    lines = [
        "-----BEGIN RSA PRIVATE KEY-----",
        "Proc-Type: 4,ENCRYPTED",
        "DEK-Info: AES-128-CBC,0123456789ABCDEF0123456789ABCDEF",
        "",
        pem(lines=1)[1][0],
        "-----END RSA PRIVATE KEY-----",
        "after",
    ]
    assert window(lines, 3) == [(3, PRIVATE_KEY_MARKER), (7, "after")]
    assert window(lines, 4) == [(4, PRIVATE_KEY_MARKER), (7, "after")]


def test_a_public_key_block_and_a_certificate_are_shown_in_a_window():
    for kind in ("PUBLIC KEY", "CERTIFICATE"):
        block, _ = pem(kind)
        assert window(block, 2, 2) == [(2, block[1]), (3, block[2])]


# ---- Grep's lines, found in the file as a stream


def test_the_lines_of_a_block_are_found_in_a_file(tmp_path):
    lines, _ = file_with_a_key()
    target = tmp_path / "a.txt"
    target.write_text("\n".join(lines) + "\n")
    assert masked_line_numbers(REDACTOR, str(target), upto=100) == frozenset(range(3, 11))


def test_a_file_with_no_key_has_no_masked_lines(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("plain\n" * 50)
    assert masked_line_numbers(REDACTOR, str(target), upto=100) == frozenset()


def test_nothing_is_found_when_redaction_is_off(tmp_path):
    lines, _ = file_with_a_key()
    target = tmp_path / "a.txt"
    target.write_text("\n".join(lines) + "\n")
    assert masked_line_numbers(Redactor(enabled=False), str(target), upto=100) == frozenset()


def test_a_block_that_begins_before_the_last_line_needed_is_followed_to_its_end(tmp_path):
    """The block is read to its end: what was asked about is line 5, in the middle of it."""
    lines, _ = file_with_a_key()
    target = tmp_path / "a.txt"
    target.write_text("\n".join(lines) + "\n")
    assert 5 in masked_line_numbers(REDACTOR, str(target), upto=5)
    assert 10 in masked_line_numbers(REDACTOR, str(target), upto=5)


def test_reading_stops_after_the_last_line_needed_when_no_block_is_open():
    seen = []

    def lines() -> Iterator[str]:
        for number in range(1, 1000):
            seen.append(number)
            yield f"line {number}"

    assert _scan(lines(), REDACTOR, 5) == set()
    assert max(seen) <= 6


def test_a_block_open_at_the_last_line_needed_is_read_to_where_the_next_begins_and_no_further():
    seen = []

    def lines() -> Iterator[str]:
        yield "-----BEGIN RSA PRIVATE KEY-----"
        yield "some text"
        yield "-----BEGIN RSA PRIVATE KEY-----"  # line 3: the first block ends before it
        for number in range(4, 1000):
            seen.append(number)
            yield "x"

    assert _scan(lines(), REDACTOR, 2) == {1}
    assert not seen, "read past the BEGIN that closed the block at the last line needed"


def test_a_block_that_has_no_end_is_followed_to_the_end_of_the_file(tmp_path):
    block, _ = pem(lines=4)
    target = tmp_path / "a.txt"
    target.write_text("\n".join(["head", *block[:-1], "tail"]) + "\n")
    assert masked_line_numbers(REDACTOR, str(target), upto=3) >= frozenset({2, 3, 4, 5, 6})


def test_a_file_is_streamed_so_that_what_is_held_does_not_grow_with_it(tmp_path):
    """Eight megabytes of lines with a key near the end: what is held at any one time is the
    line being read and the block, not the file."""
    block, _ = pem(lines=4)
    target = tmp_path / "big.txt"
    with target.open("w") as stream:
        for i in range(250_000):
            stream.write(f"line {i:06d} of a file with a good many short lines in it\n")
        stream.write("\n".join(block) + "\n")
        stream.write("last\n")
    assert target.stat().st_size > 8_000_000
    tracemalloc.start()
    try:
        found = masked_line_numbers(REDACTOR, str(target), upto=300_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert found == frozenset(range(250_001, 250_007))
    assert peak < 1_000_000, (
        f"{peak:,} bytes held while streaming a file of {target.stat().st_size:,}"
    )


def test_a_file_with_a_very_long_line_is_scanned_without_trouble(tmp_path):
    target = tmp_path / "long.txt"
    target.write_text("x" * 3_000_000 + "\n" + "y\n")
    assert masked_line_numbers(REDACTOR, str(target), upto=10) == frozenset()


def test_bytes_that_are_not_text_do_not_stop_the_scan(tmp_path):
    block, _ = pem(lines=2)
    target = tmp_path / "mixed.txt"
    target.write_bytes(b"\xff\xfe not text\n" + "\n".join(block).encode() + b"\n\x80\n")
    assert masked_line_numbers(REDACTOR, str(target), upto=10) == frozenset({2, 3, 4, 5})


@pytest.mark.parametrize("separator", ["\x0c", "\x0b", "\x1c", "\x85", chr(0x2028), "\r"])
def test_a_line_separator_the_fallback_counts_and_ripgrep_does_not_shifts_nothing_unmasked(
    tmp_path, separator
):
    """Ripgrep numbers lines by ``\\n``; the pure-Python search splits on more. Whichever
    produced the matches, every line of the block (BEGIN to END) is in the set under its own
    numbering."""
    block, _ = pem(lines=3)
    text = f"a{separator}b\n" + "\n".join(block) + "\nz\n"
    target = tmp_path / "odd.txt"
    target.write_text(text, newline="")
    found = masked_line_numbers(REDACTOR, str(target), upto=100)

    def lines_of_the_block(lines: list[str]) -> set[int]:
        return {n for n, line in enumerate(lines, 1) if any(part in line for part in block)}

    by_newline = lines_of_the_block(text.split("\n"))
    by_splitlines = lines_of_the_block(text.splitlines())
    assert by_newline != by_splitlines, "the separator does not shift the numbering: no test"
    assert by_newline <= found and by_splitlines <= found, (found, by_newline, by_splitlines)


def test_a_file_whose_only_line_ends_are_newlines_is_read_once(tmp_path, monkeypatch):
    """Crlf is the same line either way, and so is a file with nothing but newlines in it."""

    def must_not_run(_path: str) -> Iterator[str]:
        raise AssertionError("read a second time, by the other numbering")

    monkeypatch.setattr("nanoclaude.tools.keyblocks._split_lines", must_not_run)
    block, _ = pem(lines=3)
    for newline in ("\n", "\r\n"):
        target = tmp_path / f"{len(newline)}.txt"
        target.write_text(newline.join(["a", *block, "z"]) + newline, newline="")
        assert masked_line_numbers(REDACTOR, str(target), upto=100) == frozenset(range(2, 7))


def test_a_file_with_another_line_break_is_read_a_second_time_by_the_other_numbering(
    tmp_path, monkeypatch
):
    from nanoclaude.tools import keyblocks

    calls: list[str] = []
    real = keyblocks._split_lines

    def counting(path: str) -> Iterator[str]:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(keyblocks, "_split_lines", counting)
    target = tmp_path / "odd.txt"
    target.write_text("a\x0cb\nc\n", newline="")
    masked_line_numbers(REDACTOR, str(target), upto=10)
    assert calls == [str(target)]


def test_nothing_is_read_when_redaction_is_off(tmp_path, monkeypatch):
    def must_not_run(_path: str) -> Iterator[str]:
        raise AssertionError("a file was read for nothing")

    monkeypatch.setattr("nanoclaude.tools.keyblocks._physical_lines", must_not_run)
    assert masked_line_numbers(Redactor(enabled=False), str(tmp_path / "x"), upto=10) == frozenset()


def test_a_block_on_one_line_is_not_followed_past_that_line(tmp_path):
    """A key in a JSON string is complete where it stands: what follows is not held to see
    whether it belongs to it."""
    block, _ = pem(lines=3)
    target = tmp_path / "one.txt"
    with target.open("w") as stream:
        stream.write('{"key": "' + "\\n".join(block) + '"}\n')
        for i in range(250_000):
            stream.write(f"line {i:06d} of a file with a good many short lines in it\n")
    tracemalloc.start()
    try:
        found = masked_line_numbers(REDACTOR, str(target), upto=300_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert found == frozenset({1})
    assert peak < 1_000_000, f"{peak:,} bytes held"


def test_a_complete_block_and_an_open_one_on_one_line_leave_the_second_open(tmp_path):
    first, _ = pem(lines=2)
    second, _ = pem(lines=3)
    line = f"{first[0]} x {first[-1]} {second[0]}"
    target = tmp_path / "two.txt"
    target.write_text("\n".join([line, *second[1:], "after"]) + "\n")
    assert masked_line_numbers(REDACTOR, str(target), upto=100) == frozenset(range(1, 6))


# ---- the stream finds what the whole file would

_BEGIN_RSA = "-----BEGIN RSA PRIVATE KEY-----"
POOL = [
    "plain text",
    "x = 1",
    "",
    "-----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "-----BEGIN PGP PRIVATE KEY BLOCK-----",
    "-----END RSA PRIVATE KEY-----",
    "-----END PRIVATE KEY-----",
    "-----END OPENSSH PRIVATE KEY-----",
    "-----END PGP PRIVATE KEY BLOCK-----",
    "-----BEGIN PUBLIC KEY-----",
    "-----END PUBLIC KEY-----",
    "Proc-Type: 4,ENCRYPTED",
    "DEK-Info: AES-128-CBC,0123456789ABCDEF0123456789ABCDEF",
    "  MIIEowIBAAKCAQEAu1SU1LfVLPHCozMxH2Mo4lgOEePzNm0tRgeLezV6ffAt0gun",
    "# VBQ2wj7oZrKcWUhsZw5Fs5yhgUlZ5TUjDp0pFXmUYXFpZLGhWRwYZRP2tTLR8lWq",
    "ZXJzaW9uIDIgb2YgdGhlIExpY2Vuc2U=",
    'key = "-----BEGIN RSA PRIVATE KEY-----',
    '-----END RSA PRIVATE KEY-----" # tail',
    "-----BEGIN RSA PRIVATE KEY----- text -----END RSA PRIVATE KEY-----",
    "-----END RSA PRIVATE KEY----- -----BEGIN RSA PRIVATE KEY-----",
    "-----BEGIN RSA PRIVATE KEY----- x -----END RSA PRIVATE KEY----- " + _BEGIN_RSA,
    "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\\n-----END PRIVATE KEY-----",
]


def reference(lines: list[str]) -> set[int]:
    """The lines the recognizer touches when it is run over the whole file at once."""
    text = "\n".join(lines)
    starts, offset = [], 0
    for line in lines:
        starts.append(offset)
        offset += len(line) + 1
    found: set[int] = set()
    for start, end in REDACTOR.private_key_spans(text):
        first = max(i for i, s in enumerate(starts) if s <= start)
        last = max(i for i, s in enumerate(starts) if s <= max(end - 1, start))
        found.update(range(first + 1, last + 2))
    return found


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(POOL), max_size=40))
def test_the_stream_finds_the_lines_the_whole_file_would(lines):
    assert _scan(iter(lines), REDACTOR, len(lines) + 1) == reference(lines), lines


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(POOL), max_size=40), st.integers(min_value=1, max_value=40))
def test_stopping_early_never_loses_a_line_at_or_before_the_last_one_needed(lines, upto):
    found = _scan(iter(lines), REDACTOR, upto)
    assert {n for n in reference(lines) if n <= upto} <= found, lines
