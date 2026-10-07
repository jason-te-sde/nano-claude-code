"""Tests for the screen helper: what a person sees after a live display has drawn itself.

A helper that every streaming test leans on has to be shown to see what a terminal sees,
and to say so when it does not. Its cases are therefore of both kinds: output it must lay
out as a terminal does, including output a real Rich ``Live`` wrote, and output it must
refuse rather than guess at.
"""

import pytest
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from tests.cli.helpers import terminal_console
from tests.cli.screen import cursor_is_hidden, foreign_controls, on_screen


def test_text_and_line_feeds_are_laid_out_as_written():
    assert on_screen("ab\ncd") == "ab\ncd"
    # A line feed at the end leaves the cursor on a line of its own.
    assert on_screen("ab\n") == "ab\n"


def test_trailing_blanks_are_not_part_of_what_is_seen():
    assert on_screen("ab   \ncd  ") == "ab\ncd"


def test_a_carriage_return_and_an_erased_line_make_the_line_new():
    assert on_screen("hello\r\x1b[2Kbye") == "bye"


def test_a_carriage_return_alone_writes_over_the_line_it_returns_to():
    assert on_screen("hello\rHE") == "HEllo"


def test_cursor_up_and_erase_line_redraw_the_lines_above():
    # What a live display does to draw its next frame over its last.
    first = "one\ntwo"
    again = "\r\x1b[2K\x1b[1A\x1b[2K"
    assert on_screen(first + again + "three\nfour") == "three\nfour"


def test_the_cursor_cannot_go_above_the_first_line():
    assert on_screen("a\x1b[5Ab") == "ab"


def test_styling_and_the_cursor_being_shown_or_hidden_change_nothing_on_the_screen():
    assert on_screen("\x1b[?25l\x1b[1;31mbold\x1b[0m\x1b[?25h") == "bold"


def test_a_sequence_it_does_not_model_is_refused_and_not_skipped():
    # Skipped, a screen clear would leave the text it was meant to clear on the "screen",
    # and a test of "nothing was cleared" would pass for the wrong reason.
    with pytest.raises(ValueError, match="2J"):
        on_screen("text\x1b[2J")


def test_the_cursor_is_hidden_from_the_moment_it_is_hidden_until_it_is_shown():
    assert cursor_is_hidden("\x1b[?25l") is True
    assert cursor_is_hidden("\x1b[?25l text \x1b[?25h") is False
    assert cursor_is_hidden("no sequences at all") is False
    assert cursor_is_hidden("\x1b[?25l\x1b[?25h\x1b[?25l") is True


def test_a_transient_indicator_leaves_nothing_behind_when_a_real_live_display_ends():
    console, buffer = terminal_console(width=40)
    live = Live(
        Spinner("dots", text=Text("waiting for m")),
        console=console,
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    )
    live.start(refresh=True)
    assert "waiting for m" in on_screen(buffer.getvalue())
    live.stop()
    assert on_screen(buffer.getvalue()) == ""
    assert cursor_is_hidden(buffer.getvalue()) is False


def test_a_real_live_display_that_grows_ends_as_the_text_it_drew_last():
    console, buffer = terminal_console(width=40)
    live = Live(Markdown("one"), console=console, redirect_stdout=False, redirect_stderr=False)
    live.start(refresh=True)
    live.update(Markdown("one\n\ntwo"), refresh=True)
    live.update(Markdown("one\n\ntwo\n\nthree"), refresh=True)
    live.stop()
    # Three frames were drawn, each over the one before; what is left is the last, once.
    assert on_screen(buffer.getvalue()) == "one\n\ntwo\n\nthree\n"


def test_a_screen_with_a_height_scrolls_and_keeps_what_scrolled_off():
    assert on_screen("one\ntwo\nthree\nfour", height=2) == "one\ntwo\nthree\nfour"


def test_a_frame_taller_than_the_screen_leaves_copies_of_its_top_behind():
    # The cursor cannot go above the top of the screen, so a display taller than the screen
    # cannot erase the lines that scrolled off: every frame leaves its top in the scrollback.
    frame = "a\nb\nc\nd"
    redraw = "\r\x1b[2K" + "\x1b[1A\x1b[2K" * 3
    seen = on_screen(frame + redraw + frame, height=3)
    assert seen.splitlines().count("a") == 2


def test_a_frame_that_fits_the_screen_is_erased_whole_by_the_next():
    frame = "a\nb\nc"
    redraw = "\r\x1b[2K" + "\x1b[1A\x1b[2K" * 2
    assert on_screen(frame + redraw + frame, height=3) == "a\nb\nc"


def test_a_real_live_display_taller_than_the_screen_shows_the_top_and_hides_the_rest():
    # Rich's own way of fitting a display to the screen, which is why streaming does not
    # leave it to Rich: the end of a reply being written is the part that matters.
    console, buffer = terminal_console(width=40, height=5)
    text = "\n\n".join(f"line {i}" for i in range(1, 10))
    live = Live(Markdown(text), console=console, redirect_stdout=False, redirect_stderr=False)
    live.start(refresh=True)
    seen = on_screen(buffer.getvalue(), height=5)
    assert "line 1" in seen and "line 9" not in seen
    live.stop()


def test_output_of_a_live_display_holds_no_control_that_is_not_its_own():
    console, buffer = terminal_console(width=40)
    live = Live(Markdown("**bold** text"), console=console, redirect_stdout=False)
    live.start(refresh=True)
    live.stop()
    assert foreign_controls(buffer.getvalue()) == []


@pytest.mark.parametrize("injected", ["\x1b[2J", "\x1b]0;TITLE\x07", "\x07", "\x9b2J", "\x1b"])
def test_a_control_a_live_display_does_not_write_is_found_where_it_is(injected):
    found = foreign_controls(f"fine \x1b[1mbold\x1b[0m\r\x1b[2K{injected} after")
    assert found and found[0].startswith(injected[0])
