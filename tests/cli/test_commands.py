"""Tests for nanoclaude.cli.commands: the slash commands.

Spec 17.7 lists sixteen. Three describe subsystems that do not exist yet (``/agents``,
``/mcp``) or arrive in v0.2 (``/rewind``, ``/diff``, ``/checkpoints``), so they are absent
and not stubbed, and the first test asserts the v0.1 set is a subset of ``COMMANDS``: adding
the others later does not break it.
"""

from __future__ import annotations

import errno
import json
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from nanoclaude.agent.session import Session
from nanoclaude.agent.ui import Approval, AutoApprove
from nanoclaude.cli.commands import COMMANDS, dispatch
from nanoclaude.config.schema import ModelConfig
from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    user_text,
)
from nanoclaude.permissions.policy import PermissionMode, PermissionRequest, PermissionResult
from nanoclaude.providers.base import ModelError, ModelReply, Usage
from nanoclaude.testing.scripted import calls, says
from nanoclaude.testing.session import ScriptedSession, build_session
from tests.cli.helpers import plain_console


async def run_command(
    session: Session, name: str, *args: str, width: int = 100
) -> tuple[bool, str]:
    """Dispatch a command and return what it said, and whether the REPL should go on."""
    console, buffer = plain_console(width)
    carry_on = await dispatch(name, list(args), session, console)
    return carry_on, buffer.getvalue()


# --------------------------------------------------------------------------
# The set, and /help
# --------------------------------------------------------------------------


def test_every_documented_command_exists():
    """Spec 17.7 lists them; a command in the docs that does not exist is a bug."""
    expected = {
        "help",
        "clear",
        "compact",
        "status",
        "cost",
        "model",
        "mode",
        "tools",
        "resume",
        "export",
        "init",
    }
    assert expected <= set(COMMANDS)


def test_a_command_is_filed_under_its_own_name_and_says_what_it_does():
    for name, command in COMMANDS.items():
        assert command.name == name and command.summary.strip(), f"/{name} has no summary"


async def test_help_lists_every_command_with_what_it_does(tmp_repo):
    session = build_session(tmp_repo, [])
    carry_on, text = await run_command(session, "help", width=200)
    assert carry_on
    for name, command in COMMANDS.items():
        assert re.search(rf"/{name}\s+{re.escape(command.summary)}", text), f"/{name} not listed"


async def test_help_says_that_clearing_does_not_delete_the_old_conversation(tmp_repo):
    session = build_session(tmp_repo, [])
    _, text = await run_command(session, "help", width=200)
    assert "archive" in next(line for line in text.splitlines() if "/clear" in line)


async def test_an_unknown_command_suggests_help(tmp_repo):
    session = build_session(tmp_repo, [says("x")])
    carry_on, text = await run_command(session, "nope")
    assert carry_on
    assert text == "error: unknown command /nope — try /help\n"


async def test_a_command_name_is_data_and_not_markup(tmp_repo):
    session = build_session(tmp_repo, [])
    _, text = await run_command(session, "[/]")
    assert text == "error: unknown command /[/] — try /help\n"


async def test_exit_is_the_one_command_that_ends_the_repl(tmp_repo):
    session = build_session(tmp_repo, [])
    for name in COMMANDS:
        carry_on, _ = await run_command(session, name)
        assert carry_on is (name != "exit"), f"/{name}"


# --------------------------------------------------------------------------
# /clear
# --------------------------------------------------------------------------


async def test_clear_resets_the_transcript_but_keeps_the_session(tmp_repo):
    session = build_session(tmp_repo, [says("one")])
    await session.run("hi")
    session_id = session.session_id
    await run_command(session, "clear")
    assert len(session.state.transcript.messages) == 0
    assert session.session_id == session_id


async def test_clear_is_stored_at_once_so_a_resume_does_not_bring_the_conversation_back(
    tmp_repo,
):
    session = build_session(tmp_repo, [says("one")])
    await session.run("hi")
    await run_command(session, "clear")
    assert session.store is not None
    # The live messages are gone and the old ones moved to the archive, not deleted.
    assert session.store.load_transcript(session.session_id).messages == ()
    archived = session.store.db.execute(
        "SELECT COUNT(*) FROM messages_archive WHERE session_id = ?", (session.session_id,)
    ).fetchone()[0]
    assert archived == 2
    resumed = build_session(tmp_repo, [], session_id=session.session_id, resume=True)
    assert resumed.state.transcript.messages == ()


# --------------------------------------------------------------------------
# /compact
# --------------------------------------------------------------------------


async def talk(session: Session, rounds: int) -> None:
    for number in range(rounds):
        await session.follow_up(f"question {number}")


def long_answers(count: int) -> list[ModelReply]:
    return [says(f"answer {number} " + "word " * 400) for number in range(count)]


async def test_compact_summarises_the_older_turns_and_says_how_the_context_shrank(tmp_repo):
    session = build_session(
        tmp_repo,
        long_answers(4),
        compact_script=[says("SUMMARY of the work")],
        keep_recent_turns=1,
    )
    await talk(session, 4)
    before, _ = await session.context_usage()
    _, text = await run_command(session, "compact")
    after, _ = await session.context_usage()
    assert after < before
    assert text == f"compacted — context was {before} tokens, now {after}\n"
    assert (
        session.state.transcript.messages[0]
        .text()
        .startswith("[earlier conversation, compacted]\nSUMMARY of the work")
    )


async def test_compact_with_nothing_older_than_the_recent_turns_says_so(tmp_repo):
    session = build_session(tmp_repo, [says("only answer")], compact_script=[])
    await talk(session, 1)
    _, text = await run_command(session, "compact")
    assert text.startswith("nothing to compact")
    assert session.compact_model is not None and session.compact_model.requests == []


async def test_compact_passes_what_to_keep_to_the_summary(tmp_repo):
    session = build_session(
        tmp_repo,
        long_answers(4),
        compact_script=[says("SUMMARY")],
        keep_recent_turns=1,
    )
    await talk(session, 4)
    await run_command(session, "compact", "the", "auth", "module")
    assert session.compact_model is not None
    [request] = session.compact_model.requests
    assert "Pay particular attention to: the auth module" in request.transcript.messages[0].text()


# --------------------------------------------------------------------------
# /status and /cost
# --------------------------------------------------------------------------


async def test_status_reports_every_role_and_the_mode(tmp_repo):
    session = build_session(tmp_repo, [says("x")])
    _, text = await run_command(session, "status")
    assert "main" in text and "explore" in text and "default" in text


async def test_status_shows_the_context_figure_the_session_itself_reports(tmp_repo):
    # context_usage() is a coroutine: a handler that forgot to await it would hand the
    # panel a coroutine object and no figure at all.
    session = build_session(tmp_repo, [says("x")])
    await session.follow_up("hello")
    used, available = await session.context_usage()
    _, text = await run_command(session, "status")
    assert f"{used}/{available} tokens" in text


async def test_cost_shows_a_dash_for_a_model_with_no_price(tmp_repo):
    session = build_session(tmp_repo, [says("x")])
    session.router.record("main", Usage(input_tokens=10), "openai_compat", "mystery")
    _, text = await run_command(session, "cost")
    main = next(line for line in text.splitlines() if re.match(r"\W*main\W", line))
    assert re.search(r"mystery\W+10\W+0\W+0\W+-\W*$", main)
    assert "$0.00" not in text


async def test_cost_says_it_covers_this_run(tmp_repo):
    session = build_session(tmp_repo, [])
    _, text = await run_command(session, "cost")
    assert "this run" in text


# --------------------------------------------------------------------------
# /model
# --------------------------------------------------------------------------


def two_models(
    tmp_repo: Path,
    main_script: Sequence[ModelReply | ModelError],
    other_script: Sequence[ModelReply | ModelError],
) -> ScriptedSession:
    """Alias ``m`` (Sonnet) answers the main role, alias ``c`` (Haiku) is the other model."""
    return build_session(tmp_repo, main_script, compact_script=other_script)


async def test_model_lists_the_models_and_marks_the_one_the_main_role_uses(tmp_repo):
    session = two_models(tmp_repo, [], [])
    _, text = await run_command(session, "model")
    assert text == "* m: anthropic/claude-sonnet-5\n  c: anthropic/claude-haiku-4-5\n"


async def test_the_reply_after_model_comes_from_the_new_models_client_and_is_billed_to_it(
    tmp_repo,
):
    # Router keeps its own config and decides which client answers: replacing the session's
    # config alone would change what /status shows and not who answers.
    session = two_models(
        tmp_repo,
        [says("from the old model", input_tokens=7)],
        [says("from the new model", input_tokens=11, output_tokens=3)],
    )
    carry_on, text = await run_command(session, "model", "c")
    assert carry_on and text == "main role now uses c\n"
    done = await session.follow_up("hello")
    assert done.text == "from the new model"
    assert session.model.requests == []  # the old model was never asked
    assert session.compact_model is not None and len(session.compact_model.requests) == 1
    billed = session.router.by_role()["main"]
    assert (billed.model, billed.usage) == ("claude-haiku-4-5", Usage(11, 3))


async def test_model_keeps_the_sessions_config_in_step_with_the_router(tmp_repo):
    session = two_models(tmp_repo, [], [])
    await run_command(session, "model", "c")
    assert session.router.config.roles.main == "c"
    assert session.config.roles.main == "c"
    # And only the main role moved.
    assert session.config.roles.explore == "m" and session.router.config.roles.explore == "m"


async def test_model_leaves_the_rest_of_the_sessions_config_alone(tmp_repo):
    # A front end may have changed the session's own limits since the router was built.
    session = two_models(tmp_repo, [], [])
    session.config = replace(session.config, limits=replace(session.config.limits, max_turns=3))
    await run_command(session, "model", "c")
    assert session.config.limits.max_turns == 3


async def test_model_with_an_unknown_alias_changes_nothing_and_names_the_ones_that_exist(tmp_repo):
    session = two_models(tmp_repo, [], [])
    _, text = await run_command(session, "model", "[nope]")
    assert text == "error: no model named '[nope]' — choose one of: c, m\n"
    assert session.router.config.roles.main == "m" and session.config.roles.main == "m"


async def test_model_that_cannot_be_used_is_refused_with_the_reason_and_changes_nothing(tmp_repo):
    session = two_models(tmp_repo, [], [])
    session.router.config.models["keyless"] = ModelConfig(
        "anthropic", "claude-opus-5", api_key_env="NO_SUCH_KEY"
    )
    _, text = await run_command(session, "model", "keyless", width=200)
    assert text == (
        "error: role 'main' uses model 'keyless', which needs NO_SUCH_KEY "
        "— set it, or run: ncc init\n"
    )
    assert session.router.config.roles.main == "m" and session.config.roles.main == "m"


# --------------------------------------------------------------------------
# /mode
# --------------------------------------------------------------------------


async def test_mode_rejects_an_unknown_mode_and_lists_the_real_ones(tmp_repo):
    session = build_session(tmp_repo, [says("x")])
    _, text = await run_command(session, "mode", "turbo", width=200)
    assert text == (
        "error: unknown mode 'turbo' — choose one of default, plan, accept-edits "
        "(bypass is set at startup, with --dangerously-skip-permissions)\n"
    )
    assert session.policy.mode is PermissionMode.DEFAULT


async def test_mode_without_an_argument_says_which_mode_this_is(tmp_repo):
    session = build_session(tmp_repo, [])
    _, text = await run_command(session, "mode", width=200)
    assert text == "mode: default (choose default, plan or accept-edits)\n"


async def test_bypass_cannot_be_entered_from_inside_a_session(tmp_repo):
    # Spec 6.2: bypass needs --dangerously-skip-permissions, which is a flag with a long
    # spelling so that nobody types it by accident.
    session = build_session(tmp_repo, [])
    _, text = await run_command(session, "mode", "bypass", width=200)
    assert text == (
        "error: bypass mode is set at startup — restart with: ncc --dangerously-skip-permissions\n"
    )
    assert session.policy.mode is PermissionMode.DEFAULT


class CountingUI(AutoApprove):
    """Approves everything, and counts how often it was asked."""

    def __init__(self) -> None:
        self.asked = 0

    async def confirm(
        self, _call: ToolUseBlock, _request: PermissionRequest, _result: PermissionResult
    ) -> Approval:
        self.asked += 1
        return Approval.ONCE


async def test_mode_changes_what_the_executor_asks_about_and_not_only_what_status_shows(tmp_repo):
    ui = CountingUI()
    session = build_session(
        tmp_repo,
        [
            calls("Write", {"path": "a.txt", "content": "1"}, call_id="w1"),
            says("done once"),
            calls("Write", {"path": "b.txt", "content": "2"}, call_id="w2"),
            says("done twice"),
        ],
        ui=ui,
    )
    await session.follow_up("write a")
    assert ui.asked == 1  # in default mode a write has to be confirmed
    carry_on, text = await run_command(session, "mode", "accept-edits")
    assert carry_on and text == "permission mode: accept-edits\n"
    assert session.policy.mode is PermissionMode.ACCEPT_EDITS
    await session.follow_up("write b")
    assert ui.asked == 1  # and in accept-edits mode it is not
    assert (tmp_repo / "b.txt").read_text() == "2"


# --------------------------------------------------------------------------
# /tools, /resume, /init
# --------------------------------------------------------------------------


async def test_tools_lists_every_tool_and_marks_the_ones_that_cannot_change_anything(tmp_repo):
    session = build_session(tmp_repo, [])
    _, text = await run_command(session, "tools")
    lines = text.splitlines()
    assert [line.split()[0] for line in lines] == sorted(tool.name for tool in session.registry)
    assert next(line for line in lines if line.startswith("Read")).endswith("(read-only)")
    assert not next(line for line in lines if line.startswith("Edit")).endswith("(read-only)")


async def test_resume_and_init_say_where_they_live_instead_of_pretending(tmp_repo):
    session = build_session(tmp_repo, [])
    _, resume = await run_command(session, "resume")
    _, init = await run_command(session, "init")
    assert resume == "error: /resume is a startup flag — restart with: ncc --resume <id>\n"
    assert init == "error: /init is a subcommand — run: ncc init\n"


# --------------------------------------------------------------------------
# /export
# --------------------------------------------------------------------------


async def test_export_writes_markdown_to_a_file(tmp_repo):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    target = tmp_repo / "out.md"
    _, text = await run_command(session, "export", str(target), width=200)
    assert text == f"written to {target}\n"
    assert target.read_text() == (
        f"# Session {session.session_id}\n\n## user\n\nhi\n\n## assistant\n\ndone\n"
    )


async def test_export_without_a_name_writes_into_the_project(tmp_repo):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    await run_command(session, "export")
    assert (tmp_repo / f"session-{session.session_id}.md").is_file()


async def test_a_relative_export_path_is_relative_to_the_project(
    tmp_repo, tmp_path_factory, monkeypatch
):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    monkeypatch.chdir(elsewhere)
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    await run_command(session, "export", "notes.md")
    assert (tmp_repo / "notes.md").is_file() and not (elsewhere / "notes.md").exists()


async def test_export_shows_tool_calls_and_results_and_fences_them_safely(tmp_repo):
    session = build_session(tmp_repo, [calls("Read", {"path": "a.py"}, call_id="r1"), says("ok")])
    (tmp_repo / "a.py").write_text("print('x')\n```\nmore\n")
    await session.run("read it")
    target = tmp_repo / "out.md"
    await run_command(session, "export", str(target))
    exported = target.read_text()
    assert '**tool call** `Read`\n\n```json\n{"path": "a.py"}\n```' in exported
    # The file holds a fence of its own, so the one around it has to be longer.
    assert "**tool result**\n\n````\n" in exported and "\n````\n" in exported


async def test_export_leaves_out_the_models_private_reasoning_and_does_not_choke_on_it(tmp_repo):
    session = build_session(tmp_repo, [])
    session.state = replace(
        session.state,
        transcript=Transcript(
            (
                user_text("question"),
                Message(
                    "assistant", (ThinkingBlock("my private reasoning", "sig"), TextBlock("answer"))
                ),
            )
        ),
    )
    target = tmp_repo / "out.md"
    await run_command(session, "export", str(target))
    assert "answer" in target.read_text() and "private reasoning" not in target.read_text()


async def test_export_marks_a_failed_call_and_shortens_a_very_long_result(tmp_repo):
    session = build_session(tmp_repo, [])
    session.state = replace(
        session.state,
        transcript=Transcript(
            (
                user_text("go"),
                Message("assistant", (ToolUseBlock("t1", "Bash", {"command": "make"}),)),
                Message("user", (ToolResultBlock("t1", "x" * 5000, is_error=True),)),
            )
        ),
    )
    target = tmp_repo / "out.md"
    await run_command(session, "export", str(target))
    exported = target.read_text()
    assert "**tool result** (error)" in exported
    assert "x" * 2000 in exported and "x" * 2001 not in exported
    assert "[3000 more characters not exported]" in exported


#: A clear, a title, the 8-bit forms of both, DEL, the two ends of the C1 block, a bell and a
#: carriage return.
HOSTILE = "a\x1b[2J\x1b]0;T\x07\x9b2J\x9d0;T\x9c\x7f\x80\x85\x9f\r b"

#: Every control a terminal acts on, except tab and newline.
CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


async def exported(
    tmp_repo: Path,
    *,
    session_id: str = "s1",
    user: str = "u",
    text: str = "t",
    name: str = "Read",
    arguments: dict[str, object] | None = None,
    result: str = "r",
) -> str:
    """What ``/export`` writes for a conversation made of exactly these pieces."""
    session = build_session(tmp_repo, [], session_id=session_id)
    call = ToolUseBlock("t1", name, arguments if arguments is not None else {"path": "p"})
    session.state = replace(
        session.state,
        transcript=Transcript(
            (
                user_text(user),
                Message("assistant", (TextBlock(text), call)),
                Message("user", (ToolResultBlock("t1", result),)),
            )
        ),
    )
    target = tmp_repo / "out.md"
    await run_command(session, "export", str(target))
    return target.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "piece",
    ["session_id", "user", "text", "name", "key", "value", "result"],
)
async def test_an_exported_file_cannot_act_on_the_terminal_of_whoever_prints_it(tmp_repo, piece):
    # The file is read by something that prints it, "cat" in a terminal included. Whatever the
    # model or a tool put in a conversation must not reach it as a character the terminal acts
    # on, in any place the export writes it: the id, both sides' text, a call's name, the
    # key and the value of its arguments, and a result. DEL and C1 are the ones JSON does not
    # escape.
    # Only the piece under test holds the hostile text, so that each case fails for its own
    # place and not for another's.
    pieces: dict[str, object] = {}
    if piece == "key":
        pieces["arguments"] = {HOSTILE: "p"}
    elif piece == "value":
        pieces["arguments"] = {"path": HOSTILE}
    else:
        pieces[piece] = HOSTILE
    written = await exported(tmp_repo, **pieces)  # type: ignore[arg-type]
    assert not CONTROLS.search(written), repr(CONTROLS.findall(written))
    if piece in ("key", "value"):
        # Escaped, not stripped: JSON keeps what was sent.
        assert "\\u009b2J\\u009d0;T\\u009c\\u007f" in written
    else:
        # Stripped: what is left of the text is still there.
        assert "a2J0;T b" in written


async def test_the_arguments_of_a_call_are_exported_as_json_that_reads_back_unchanged(tmp_repo):
    # DEL and C1 are escaped as \u00XX, which JSON allows anywhere in a string: the file stays
    # valid, and nothing is lost.
    arguments: dict[str, object] = {
        "path": "a.py",
        HOSTILE: HOSTILE,
        "n": 3,
        "nested": {"k": ["\x7f", "\x9b", "\x9f", "caf\N{LATIN SMALL LETTER E WITH ACUTE}"]},
    }
    written = await exported(tmp_repo, arguments=arguments)
    block = re.search(r"```json\n(.*?)\n```", written, re.S)
    assert block is not None and json.loads(block.group(1)) == arguments
    assert "\\u007f" in written and "\\u009b" in written
    # Only what a terminal acts on is escaped: ordinary text, and letters beyond ASCII, are not.
    assert "caf\N{LATIN SMALL LETTER E WITH ACUTE}" in written and "a.py" in written


async def test_export_will_not_overwrite_a_file_that_is_there(tmp_repo):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    target = tmp_repo / "out.md"
    target.write_text("precious\n")
    _, text = await run_command(session, "export", str(target), width=200)
    assert text == f"error: {target} already exists — give another name, or remove it first\n"
    assert target.read_text() == "precious\n"


async def test_export_to_a_directory_that_is_not_there_is_an_error_line_not_a_traceback(tmp_repo):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    target = tmp_repo / "missing" / "out.md"
    _, text = await run_command(session, "export", str(target), width=300)
    assert text.startswith(f"error: could not write {target} — ") and text.endswith("\n")


async def test_a_tilde_names_the_home_directory(tmp_repo, tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    _, text = await run_command(session, "export", "~/out.md", width=300)
    assert (home / "out.md").is_file() and text == f"written to {home / 'out.md'}\n"
    assert not (tmp_repo / "~").exists()


async def test_a_home_directory_nobody_has_is_an_error_line_and_not_a_bug(tmp_repo):
    # Path.expanduser raises RuntimeError for ~name when there is no such user.
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    _, text = await run_command(session, "export", "~no_such_user_zz9/out.md", width=300)
    assert text == (
        "error: cannot expand '~no_such_user_zz9/out.md' \u2014 it starts with ~ but no home "
        "directory is known for it; write the full path\n"
    )


async def test_an_exported_file_is_private_to_whoever_made_it(tmp_repo):
    # It holds the whole conversation, tool output included.
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    target = tmp_repo / "out.md"
    await run_command(session, "export", str(target))
    assert stat.S_IMODE(target.stat().st_mode) & 0o077 == 0


async def test_text_that_is_not_valid_unicode_is_exported_with_it_spelled_out(tmp_repo):
    # A lone surrogate cannot be encoded as UTF-8. It used to be found out after the file
    # had been created, which left an empty file and made the next export say it was there.
    session = build_session(tmp_repo, [says("a" + chr(0xD800) + "b")])
    await session.run("hi")
    target = tmp_repo / "out.md"
    _, text = await run_command(session, "export", str(target), width=300)
    assert text == f"written to {target}\n"
    assert "a\\ud800b" in target.read_text(encoding="utf-8")


async def test_a_write_that_fails_leaves_no_file_to_block_the_next_export(tmp_repo, monkeypatch):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    target = tmp_repo / "out.md"
    real_fdopen = os.fdopen

    class FullDisk:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def __enter__(self) -> FullDisk:
            return self

        def __exit__(self, *_exc: object) -> None:
            self._handle.close()  # type: ignore[attr-defined]

        def write(self, _data: bytes) -> int:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    with monkeypatch.context() as patched:
        patched.setattr(os, "fdopen", lambda fd, *a, **k: FullDisk(real_fdopen(fd, *a, **k)))
        _, text = await run_command(session, "export", str(target), width=300)
    assert text == f"error: could not write {target} \u2014 No space left on device\n"
    assert not target.exists()
    _, text = await run_command(session, "export", str(target), width=300)
    assert text == f"written to {target}\n" and target.is_file()


async def test_a_name_with_a_null_byte_in_it_is_an_error_line_and_not_a_bug(tmp_repo):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    _, text = await run_command(session, "export", "a\x00b", width=300)
    # The reason is the interpreter's own: 3.11 says "embedded null byte", 3.13 says
    # "open: embedded null character in path". "embedded null" is what both say.
    assert text.startswith("error: could not write ") and "embedded null" in text
    assert text.count("\n") == 1


async def test_a_tilde_name_with_a_null_byte_in_it_is_an_error_line_and_not_a_bug(tmp_repo):
    # Path.expanduser looks the name up, and the lookup refuses a NUL with a ValueError of its
    # own, which is not the RuntimeError it raises for a name nobody has.
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    _, text = await run_command(session, "export", "~a\x00b/out.md", width=300)
    assert text.startswith("error: cannot expand '~a\\x00b/out.md' \u2014 ")
    assert text.count("\n") == 1
    assert not list(tmp_repo.glob("*.md"))


async def test_export_takes_the_rest_of_the_line_as_the_name(tmp_repo):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hi")
    await run_command(session, "export", "my", "notes.md")
    assert (tmp_repo / "my notes.md").is_file()


@pytest.fixture
def bracketed_session(tmp_repo: Path) -> ScriptedSession:
    """A session whose id and model name hold what rich reads as markup."""
    session = build_session(tmp_repo, [], session_id="[id]-[/]")
    session.router.config.models["m"] = ModelConfig("anthropic", "model-[x]")
    return session


@pytest.mark.parametrize("name", sorted(COMMANDS))
async def test_no_command_trips_over_a_bracket_in_what_it_prints(bracketed_session, name):
    await run_command(bracketed_session, name)  # raises MarkupError if anything is parsed


async def test_the_names_a_command_prints_come_out_as_they_were_written(bracketed_session):
    _, status = await run_command(bracketed_session, "status", width=200)
    _, models = await run_command(bracketed_session, "model")
    assert "model-[x] (m)" in status and "[id]-[/]" in status
    assert models.startswith("* m: anthropic/model-[x]\n")
