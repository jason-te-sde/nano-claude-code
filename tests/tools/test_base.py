import os
from collections.abc import Hashable
from pathlib import Path

import pytest

from nanoclaude.tools.base import (
    Tool,
    ToolArgumentError,
    ToolOutcome,
    optional_int,
    require_str,
    sanitize,
)
from nanoclaude.tools.read import ReadTool


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("\x1b[31mred\x1b[0m", "red"),  # SGR colour
        ("\x1b[2J\x1b[H cleared", " cleared"),  # clear screen + home
        ("\x1b]0;window title\x07done", "done"),  # OSC title
        ("bell\x07", "bell"),
        ("back\x08space", "backspace"),
        ("carriage\rreturn", "carriagereturn"),
        ("keep\ttabs\nand newlines", "keep\ttabs\nand newlines"),
        ("plain text", "plain text"),
    ],
)
def test_control_sequences_are_stripped_from_tool_output(raw, expected):
    """Review Focus #4. git log --color and most test runners emit these.

    Into the terminal they can clear the screen or move the cursor; into the
    transcript they are tokens that mean nothing to the model.
    """
    assert sanitize(raw) == expected


@pytest.mark.parametrize("code", range(0x80, 0xA0), ids=lambda code: f"U+{code:04X}")
def test_every_c1_control_is_stripped(code):
    # The 8-bit forms of the escape sequences: U+009B is a CSI, U+009D an OSC, U+0090 a
    # DCS and U+009C the terminator of a string. A terminal that honours C1 in UTF-8
    # acts on each as it would on the two-byte form, so they go with the rest.
    assert sanitize(f"a{chr(code)}b") == "ab"


def test_the_eight_bit_forms_of_the_sequences_lose_their_introducers():
    # What is left is inert text: it does not introduce anything.
    assert sanitize("a\x9b2Jb\x9d0;T\x9cc\x90qd") == "a2Jb0;Tcqd"


def test_delete_is_stripped():
    assert sanitize("del\x7fete") == "delete"


@pytest.mark.parametrize(
    "text",
    [
        "no\u00a0break\u00a0space",  # U+00A0 is the first character after the C1 block
        "caf\u00e9 \u00ff",
        "bidi \u202a\u202b\u202c\u202d\u202e and \u2066\u2067\u2068\u2069 isolates",
        "tab\tand\nnewline",
    ],
    ids=["no-break-space", "latin-1-letters", "bidirectional-controls", "tab-and-newline"],
)
def test_what_a_terminal_does_not_act_on_is_left_alone(text):
    # The bidirectional controls do not act on a terminal, and file contents may hold
    # them: stripping them from what Read returns would leave Edit unable to match
    # the lines that have them.
    assert sanitize(text) == text


def test_require_str_names_the_argument_and_the_type_it_got():
    with pytest.raises(ToolArgumentError, match="path must be a string, got int"):
        require_str({"path": 3}, "path")


def test_optional_int_rejects_booleans_because_bool_is_an_int_in_python():
    with pytest.raises(ToolArgumentError, match="must be an integer, got bool"):
        optional_int({"limit": True}, "limit", 10, minimum=1)


def test_optional_int_enforces_its_minimum():
    with pytest.raises(ToolArgumentError, match="at least 1"):
        optional_int({"offset": 0}, "offset", 1, minimum=1)


def test_resolving_an_empty_path_is_rejected_by_the_context_not_the_sandbox(ctx):
    """Sandbox.resolve("") raises the same ValueError text (sandbox.py), but as a
    bare ValueError, not a ToolArgumentError. If ToolContext.resolve's own guard
    were deleted, the call below would still raise -- just the wrong type --
    and this match on ToolArgumentError specifically would stop catching it.
    """
    with pytest.raises(ToolArgumentError, match="path must not be empty"):
        ctx.resolve("")


def test_display_shows_a_path_inside_root_relative_to_it(ctx, tmp_repo):
    resolved = ctx.resolve("src/a.py")
    assert ctx.display(resolved) == str(Path("src") / "a.py")


def test_display_shows_a_path_outside_root_unchanged(ctx):
    assert ctx.display("/etc/passwd") == "/etc/passwd"


def test_read_tool_satisfies_the_tool_protocol():
    """Tool is the one interface every registered tool must meet (registry.py)."""
    assert isinstance(ReadTool(), Tool)


def test_file_related_dataclasses_are_hashable():
    """FileStamp, FileSnapshot and ToolOutcome hold only primitives and tuples of
    primitives (see fs.py), so, unlike ToolContext below, frozen=True's default
    __hash__ is safe to leave in place here.
    """
    outcome = ToolOutcome("t1", "content", is_error=False, observed=())
    assert isinstance(outcome, Hashable)
    hash(outcome)  # must not raise


def test_tool_context_is_declared_unhashable(ctx):
    # read_state is a mapping, so the class declares __hash__ = None instead of
    # inheriting a generated hash that would raise naming "dict".
    assert not isinstance(ctx, Hashable)
    with pytest.raises(TypeError, match="ToolContext"):
        hash(ctx)


@pytest.mark.parametrize("value", ["fifty", 2.5, None, [1]])
def test_optional_int_refuses_a_value_that_is_not_an_integer(value):
    with pytest.raises(ToolArgumentError, match="offset must be an integer, got"):
        optional_int({"offset": value}, "offset", 1, minimum=1)


def test_a_name_with_bytes_that_are_not_text_is_displayed_in_a_form_that_can_be_sent(ctx, tmp_repo):
    """A name the system decoded with lone surrogates cannot be encoded as UTF-8."""
    shown = ctx.display(os.fsdecode(str(tmp_repo).encode() + b"/sub/caf\xe9.txt"))
    assert shown == "sub/caf\ufffd.txt"
    shown.encode("utf-8")


def test_a_name_that_is_text_is_displayed_as_it_is(ctx, tmp_repo):
    assert ctx.display(str(tmp_repo / "caf\u00e9.txt")) == "caf\u00e9.txt"
