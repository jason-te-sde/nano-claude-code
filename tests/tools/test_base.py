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


def test_tool_context_reports_hashable_but_hashing_it_raises(ctx):
    """Not decided the way ToolSpec/ModelRequest/ModelReply were (providers/base.py).

    read_state is typed Mapping[str, FileStamp], but every real caller -- this
    module's own `ctx` fixture included -- passes a plain dict, which is not
    hashable. frozen=True plus the default eq=True generates a real __hash__
    unconditionally, so isinstance(ctx, Hashable) reports True even though
    hash(ctx) actually raises, naming "dict" rather than ToolContext -- exactly
    the hazard providers/base.py's ToolSpec/ModelRequest/ModelReply close with
    an explicit ``__hash__ = None``. ToolContext carries no such guard in the
    brief this module was transcribed from; pinned here as a known gap rather
    than left undocumented (see the task report).
    """
    assert isinstance(ctx, Hashable)
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        hash(ctx)
