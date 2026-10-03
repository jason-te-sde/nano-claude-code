"""Tests for nanoclaude.agent.session and nanoclaude.testing.session.

The first block is the brief's own eleven tests. Two of them changed shape and the
module says why where it happens: the compaction test could not pass as written
(a one-message conversation is never compacted, so the marker it looks for never
appears), and the retry test must not wait out a real back-off.

The rest pin the notes the brief carries and what the session owns that no other
module can: that a bug in a tool does not end it, that an unknown cost is stored
as unknown, that a reply of only complaints is asked for again a bounded number of
times, that the configured retry policy is the one in force, that the context
figure is measured the way compaction measures it, that no request sets a
temperature, and that a session stays usable after Ctrl+C.

Nothing here waits for time to pass: the retry policy under test has no delay or
a delay of a millisecond, and cancellation is driven with events.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from nanoclaude.agent.loop import StopReason
from nanoclaude.agent.session import Session, new_session_id
from nanoclaude.agent.ui import AutoApprove, AutoDecline
from nanoclaude.config.schema import LimitsConfig, ModelConfig, RolesConfig
from nanoclaude.conversation.budget import (
    Budget,
    ContextTooSmallError,
    HeuristicCounter,
    transcript_text,
)
from nanoclaude.conversation.compaction import SUMMARY_TEMPLATE
from nanoclaude.conversation.store import Store
from nanoclaude.conversation.transcript import (
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    assistant_text,
    user_text,
    validate,
)
from nanoclaude.prompts import SYSTEM_PROMPT
from nanoclaude.providers.base import ModelError, ModelReply, ModelRequest, StopKind, Usage
from nanoclaude.providers.capabilities import (
    CONSERVATIVE_DEFAULT,
    Capabilities,
    capabilities_for,
)
from nanoclaude.providers.retry import RetryPolicy
from nanoclaude.providers.texttools import MAX_PARSE_RETRIES
from nanoclaude.testing.scripted import calls, calls_many, says
from nanoclaude.testing.session import ScriptedSession, build_session
from nanoclaude.tools.base import ToolContext, ToolOutcome

#: A model id neither the capability table nor the price book has heard of. Its
#: cost is unknown, and it gets the conservative capabilities unless a test gives
#: it others.
UNPRICED = "model-without-a-price"

#: A model with no native tool calling and a window the protocol prompt fits in
#: with room to spare. The brief's own tests use CONSERVATIVE_DEFAULT (8,192
#: tokens); the retry tests add messages and should not depend on how close the
#: fixed prompt sits to that window.
TEXT_ONLY = Capabilities(
    native_tools=False,
    parallel_tools=False,
    cache="none",
    context_window=32_000,
    max_output=4_000,
)

BAD_CALL = '<tool name="Read">{"path": </tool>'

#: What an unconfigured session runs on, and what the compact role runs on when it
#: is given a model of its own. Read from the table, so a test that needs a window
#: or an output limit never states a number the table may have moved on from.
SONNET = capabilities_for("anthropic", "claude-sonnet-5")
HAIKU = capabilities_for("anthropic", "claude-haiku-4-5")


def available_in(capabilities: Capabilities) -> int:
    """Spec 7.3: the window, less the room kept for the reply, less a tenth as a margin."""
    return capabilities.context_window - capabilities.max_output - capabilities.context_window // 10


@pytest.fixture(autouse=True)
def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment block asks git for the branch on every turn: two forks.

    Which branch the repository is on has no bearing on anything tested here, and
    a temp directory that happened to sit inside another repository would change
    the system prompt from machine to machine. assemble() has its own tests.
    """
    monkeypatch.setattr("nanoclaude.context.assemble.git_state", lambda _root: "")


class Watcher(AutoApprove):
    """Approves everything, and remembers what the session told the UI."""

    def __init__(self) -> None:
        self.replies: list[ModelReply] = []
        self.retries: list[tuple[int, float, str]] = []

    def on_reply(self, reply: ModelReply) -> None:
        self.replies.append(reply)

    def on_retry(self, attempt: int, delay_s: float, reason: str) -> None:
        self.retries.append((attempt, delay_s, reason))


def tool_history(rounds: int, *, result_chars: int) -> Transcript:
    """A conversation that already ran ``rounds`` tool rounds and ended on an answer.

    ``2 * rounds + 2`` messages: the request, a call and its result per round, and
    the answer. Ids and paths are ``h0``/``f0.py`` and so on.
    """
    messages = [user_text("start")]
    for i in range(rounds):
        messages.append(
            Message("assistant", (ToolUseBlock(f"h{i}", "Read", {"path": f"f{i}.py"}),))
        )
        messages.append(
            Message("user", (ToolResultBlock(f"h{i}", f"result {i}: " + "x" * result_chars),))
        )
    messages.append(assistant_text("all done"))
    return Transcript(tuple(messages))


def is_summary_request(request: ModelRequest) -> bool:
    return transcript_text(request.transcript).startswith(SUMMARY_TEMPLATE)


async def reached(event: asyncio.Event) -> None:
    """Wait for something the code under test must do, and fail if it never does.

    A bare ``event.wait()`` would hang the run when the thing never happens, which is
    exactly what a regression looks like. Nothing here takes anywhere near five
    seconds.
    """
    await asyncio.wait_for(event.wait(), timeout=5)


def tool_results(transcript: Transcript) -> list[ToolResultBlock]:
    return [block for message in transcript.messages for block in message.tool_results()]


def stored_row(session: ScriptedSession) -> Any:
    assert session.store is not None
    return session.store.db.execute(
        "SELECT * FROM sessions WHERE id = ?", (session.session_id,)
    ).fetchone()


# --------------------------------------------------------------------------
# The brief's own tests
# --------------------------------------------------------------------------


async def test_a_one_turn_conversation_completes(tmp_repo):
    session = build_session(tmp_repo, [says("all done")])
    done = await session.run("hello")
    assert done.reason is StopReason.COMPLETED and done.text == "all done"


async def test_a_tool_call_runs_and_the_result_goes_back_to_the_model(tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\n")
    session = build_session(
        tmp_repo, [calls("Read", {"path": "a.py"}, call_id="t1"), says("it sets x")]
    )
    done = await session.run("what is in a.py")
    assert done.text == "it sets x"
    assert "x = 1" in transcript_text(done.state.transcript)
    # Not only in the final state: the model's second request is what carried it.
    answered = tool_results(session.model.requests[1].transcript)
    assert [r.tool_use_id for r in answered] == ["t1"] and "x = 1" in answered[0].content


async def test_the_session_is_persisted_turn_by_turn(tmp_repo):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hello")
    assert session.store is not None
    reloaded = session.store.load_transcript(session.session_id)
    assert reloaded == session.state.transcript


async def test_a_follow_up_continues_the_same_transcript(tmp_repo):
    session = build_session(tmp_repo, [says("first"), says("second")])
    await session.run("one")
    done = await session.follow_up("two")
    assert done.text == "second"
    assert len(done.state.transcript.messages) == 4
    # The second request carried the whole first exchange, then the new prompt.
    assert [m.text() for m in session.model.requests[1].transcript.messages] == [
        "one",
        "first",
        "two",
    ]


async def test_a_retryable_provider_error_is_retried(tmp_repo):
    session = build_session(
        tmp_repo, [ModelError("overloaded", retryable=True, status=529), says("ok")]
    )
    done = await session.run("hello")
    assert done.text == "ok"
    assert len(session.model.requests) == 2


async def test_a_non_retryable_provider_error_surfaces(tmp_repo):
    session = build_session(tmp_repo, [ModelError("bad key", retryable=False, status=401)])
    with pytest.raises(ModelError, match="bad key"):
        await session.run("hello")
    assert len(session.model.requests) == 1  # asked once; a 401 is not retried


async def test_crossing_the_hard_threshold_triggers_a_full_compaction(tmp_repo):
    """The brief ran this on a one-message conversation, which full_compact leaves
    alone: with nothing older than the recent turns there is nothing to summarise.
    It needs a history longer than the turns kept verbatim."""
    session = build_session(
        tmp_repo,
        [says("SUMMARY of the work"), says("answer")],
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    done = await session.follow_up("what next")
    assert "[earlier conversation, compacted]" in transcript_text(done.state.transcript)
    assert done.text == "answer"


async def test_usage_is_attributed_to_the_main_role(tmp_repo):
    session = build_session(tmp_repo, [says("done", input_tokens=50, output_tokens=5)])
    await session.run("hello")
    main = session.router.by_role()["main"]
    assert main.usage.input_tokens == 50
    assert (main.adapter, main.model) == ("anthropic", "claude-sonnet-5")


async def test_a_model_without_native_tools_gets_the_text_protocol(tmp_repo):
    (tmp_repo / "a.py").write_text("x = 1\n")
    session = build_session(
        tmp_repo,
        [says('<tool name="Read">{"path": "a.py"}</tool>'), says("it sets x")],
        capabilities=CONSERVATIVE_DEFAULT,
    )
    done = await session.run("read a.py")
    assert done.text == "it sets x"
    assert "x = 1" in transcript_text(done.state.transcript)
    # The loop cannot tell the difference: what the transcript holds is a real call.
    assert [b.name for b in done.state.transcript.messages[1].tool_uses()] == ["Read"]


async def test_the_system_prompt_names_the_text_protocol_only_when_needed(tmp_repo):
    native = build_session(tmp_repo, [says("done")])
    await native.run("hi")
    assert "<tool name=" not in native.model.requests[0].system
    assert native.model.requests[0].tools  # sent as tools, not as prose

    fallback = build_session(tmp_repo, [says("done")], capabilities=CONSERVATIVE_DEFAULT)
    await fallback.run("hi")
    assert "<tool name=" in fallback.model.requests[0].system
    assert fallback.model.requests[0].tools == ()  # said once, in the prompt


async def test_the_turn_limit_ends_the_session_cleanly(tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    script = [calls("Read", {"path": "a.py"}, call_id=f"t{i}") for i in range(10)]
    session = build_session(tmp_repo, script, max_turns=3)
    done = await session.run("keep reading")
    assert done.reason is StopReason.TURN_LIMIT
    validate(done.state.transcript)  # closed cleanly: every call has its answer


# --------------------------------------------------------------------------
# new_session_id
# --------------------------------------------------------------------------


def test_a_session_id_is_twelve_hex_characters_and_never_repeats():
    ids = {new_session_id() for _ in range(300)}
    assert len(ids) == 300
    assert all(len(i) == 12 and set(i) <= set("0123456789abcdef") for i in ids)


# --------------------------------------------------------------------------
# Carried note: an unknown cost is stored as unknown
# --------------------------------------------------------------------------


async def test_a_priced_session_is_stored_with_what_it_cost(tmp_repo):
    session = build_session(tmp_repo, [says("done", input_tokens=1_000, output_tokens=100)])
    await session.run("hello")
    # Sonnet 5 is $2 per million input tokens and $10 per million output tokens:
    # (1,000 * 2 + 100 * 10) / 1,000,000.
    assert session.cost == pytest.approx(0.003)
    row = stored_row(session)
    assert row["total_cost_usd"] == pytest.approx(0.003)
    assert (row["total_input_tokens"], row["total_output_tokens"]) == (1_000, 100)
    assert row["ended_at"] is not None


async def test_an_unpriced_session_is_stored_with_an_unknown_cost_not_zero(tmp_repo):
    session = build_session(
        tmp_repo, [says("done", input_tokens=1_000, output_tokens=100)], model=UNPRICED
    )
    await session.run("hello")
    assert session.cost is None
    row = stored_row(session)
    assert row["total_cost_usd"] is None  # NULL, which a display shows as unknown
    assert (row["total_input_tokens"], row["total_output_tokens"]) == (1_000, 100)
    assert row["ended_at"] is not None  # the session is still recorded as finished


async def test_a_session_that_used_nothing_costs_a_known_zero_not_an_unknown(tmp_repo):
    # A priced model with no tokens spent is $0.00 and that is a fact. It must not
    # be mistaken for the unknown above by any falsy check on the way to the store.
    session = build_session(tmp_repo, [says("done")])
    await session.run("hello")
    assert session.cost == 0.0
    assert stored_row(session)["total_cost_usd"] == 0.0


async def test_a_second_unpriced_model_makes_the_whole_total_unknown(tmp_repo):
    # Priced main, unpriced compaction model: the total is a guess, so it is unknown.
    session = build_session(
        tmp_repo,
        [says("answer", input_tokens=1_000)],
        compact_script=[says("SUMMARY")],
        compact_model=UNPRICED,
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    assert session.router.by_role()["main"].cost is not None
    assert session.cost is None
    assert stored_row(session)["total_cost_usd"] is None


async def test_usage_and_cost_follow_the_router_as_the_session_goes_on(tmp_repo):
    session = build_session(
        tmp_repo,
        [
            says("one", input_tokens=10, output_tokens=1),
            says("two", input_tokens=20, output_tokens=2),
        ],
    )
    assert session.usage == Usage() and session.cost == 0.0
    await session.run("a")
    assert session.usage == Usage(10, 1)
    await session.follow_up("b")
    assert session.usage == Usage(30, 3)
    assert session.usage == session.router.total_usage()
    assert session.cost == session.router.total_cost()


# --------------------------------------------------------------------------
# Carried note: MAX_PARSE_RETRIES
# --------------------------------------------------------------------------


async def test_a_reply_of_only_complaints_is_asked_for_again_with_the_complaints_fed_back(
    tmp_repo,
):
    (tmp_repo / "a.py").write_text("x = 1\n")
    session = build_session(
        tmp_repo,
        [says(BAD_CALL), says('<tool name="Read">{"path": "a.py"}</tool>'), says("it sets x")],
        capabilities=TEXT_ONLY,
    )
    done = await session.run("read a.py")
    assert done.text == "it sets x"
    assert len(session.model.requests) == 3

    retry_request = session.model.requests[1]
    validate(retry_request.transcript)  # roles still alternate
    _prompt, attempt, correction = retry_request.transcript.messages
    # What the model said is in the transcript as it said it, not with our note
    # appended: the complaint arrives once, as the user's next message.
    assert attempt.role == "assistant" and attempt.text() == BAD_CALL
    assert correction.role == "user"
    assert "not valid JSON" in correction.text()
    assert "Read" in correction.text()
    # It says what to do next, not only what went wrong.
    assert "<tool name=" in correction.text()
    # The conversation went on normally after it.
    assert "x = 1" in transcript_text(done.state.transcript)


async def test_a_reply_of_only_complaints_is_retried_at_most_the_limit_and_then_stops(tmp_repo):
    script = [says(f"{BAD_CALL} attempt {n}") for n in range(MAX_PARSE_RETRIES + 3)]
    session = build_session(tmp_repo, script, capabilities=TEXT_ONLY)
    done = await session.run("read a.py")
    # The first ask and MAX_PARSE_RETRIES more; the rest of the script is never used.
    assert len(session.model.requests) == MAX_PARSE_RETRIES + 1
    assert not session.model.exhausted
    # It stops with the last reply, complaints and all, as a finished turn.
    assert done.reason is StopReason.COMPLETED
    assert f"attempt {MAX_PARSE_RETRIES}" in done.text
    assert "[tool protocol]" in done.text
    validate(done.state.transcript)


async def test_a_reply_with_one_valid_call_is_not_retried_for_the_malformed_one_beside_it(
    tmp_repo,
):
    (tmp_repo / "a.py").write_text("x = 1\n")
    both = '<tool name="Read">{"path": "a.py"}</tool>\n<tool name="Grep">{not json}</tool>'
    session = build_session(tmp_repo, [says(both), says("done")], capabilities=TEXT_ONLY)
    done = await session.run("go")
    # One request for the turn with the call in it and one after the tool ran.
    assert len(session.model.requests) == 2
    assert done.text == "done"
    after = session.model.requests[1].transcript
    assert [r.tool_use_id for r in tool_results(after)] == ["tt_1"]  # the call ran
    # The complaint stays in the reply for the model to see, but nobody asked it
    # to write the call again: no user message in the history is a correction.
    assert "[tool protocol]" in after.messages[1].text()
    assert [m.role for m in after.messages] == ["user", "assistant", "user"]


async def test_the_parse_retry_limit_counts_failures_in_a_row_not_in_a_session(tmp_repo):
    # Two failures, a good call, two more failures, then an answer. With a limit of
    # two in a row every reply is used; a limit of two on the whole session would
    # stop at the fourth reply.
    (tmp_repo / "a.py").write_text("x\n")
    good = '<tool name="Read">{"path": "a.py"}</tool>'
    script = [says(BAD_CALL)] * 2 + [says(good)] + [says(BAD_CALL)] * 2 + [says("finished")]
    session = build_session(tmp_repo, script, capabilities=TEXT_ONLY)
    done = await session.run("go")
    assert done.text == "finished"
    assert session.model.exhausted
    assert len(session.model.requests) == 6


async def test_every_attempt_is_billed_and_shown_even_the_ones_that_were_asked_again(tmp_repo):
    watcher = Watcher()
    script = [
        says(BAD_CALL, input_tokens=10, output_tokens=1) for _ in range(MAX_PARSE_RETRIES + 1)
    ]
    session = build_session(tmp_repo, script, capabilities=TEXT_ONLY, ui=watcher)
    await session.run("go")
    attempts = MAX_PARSE_RETRIES + 1
    assert session.router.by_role()["main"].usage == Usage(10 * attempts, attempts)
    assert len(watcher.replies) == attempts


async def test_asking_again_does_not_use_up_turns(tmp_repo):
    # Two turns allowed: one tool round and the answer. Asking again twice on the
    # way must not count against them.
    (tmp_repo / "a.py").write_text("x\n")
    good = '<tool name="Read">{"path": "a.py"}</tool>'
    script = [says(BAD_CALL), says(BAD_CALL), says(good), says("finished")]
    session = build_session(tmp_repo, script, capabilities=TEXT_ONLY, max_turns=2)
    done = await session.run("go")
    assert done.reason is StopReason.COMPLETED and done.text == "finished"
    assert done.state.turn == 1


async def test_a_model_with_native_tools_is_never_asked_again_for_a_text_call(tmp_repo):
    # Both directions of the guard: only a model without native calls is parsed
    # for text calls, so only its malformed ones are retried.
    session = build_session(tmp_repo, [says(BAD_CALL), says("never used")])
    done = await session.run("go")
    assert len(session.model.requests) == 1
    assert done.text == BAD_CALL and "[tool protocol]" not in done.text


async def test_text_protocol_calls_get_ids_that_are_unique_across_turns(tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    one = '<tool name="Read">{"path": "a.py"}</tool>'
    session = build_session(tmp_repo, [says(one), says(one), says("done")], capabilities=TEXT_ONLY)
    done = await session.run("go")
    ids = [r.tool_use_id for r in tool_results(done.state.transcript)]
    assert ids == ["tt_1", "tt_2"]
    validate(done.state.transcript)  # validate() rejects a repeated id


# --------------------------------------------------------------------------
# Carried note: a bug in one tool must not end the session
# --------------------------------------------------------------------------


async def test_a_bug_in_one_tool_does_not_end_the_session(tmp_repo, monkeypatch):
    from nanoclaude.tools.read import ReadTool

    async def explode(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        raise RuntimeError("list index out of range")

    monkeypatch.setattr(ReadTool, "run", explode)
    (tmp_repo / "a.py").write_text("x\n")
    session = build_session(
        tmp_repo, [calls("Read", {"path": "a.py"}, call_id="t1"), says("that tool is broken")]
    )
    done = await session.run("read it")
    assert done.reason is StopReason.COMPLETED and done.text == "that tool is broken"
    # The model was told what happened and could answer.
    [result] = tool_results(session.model.requests[1].transcript)
    assert result.is_error and "RuntimeError: list index out of range" in result.content
    assert "bug" in result.content
    # And the audit table says so too.
    assert session.store is not None
    row = session.store.db.execute(
        "SELECT outcome, session_id FROM tool_calls WHERE tool_use_id = 't1'"
    ).fetchone()
    assert (row["outcome"], row["session_id"]) == ("error", session.session_id)


# --------------------------------------------------------------------------
# Carried note: the retry policy and ui.on_retry
# --------------------------------------------------------------------------


async def test_a_retry_is_reported_to_the_ui_with_the_attempt_the_delay_and_the_reason(tmp_repo):
    watcher = Watcher()
    session = build_session(
        tmp_repo,
        [ModelError("overloaded", retryable=True, status=529), says("ok")],
        ui=watcher,
        retry=RetryPolicy(base_delay_s=0.001, max_delay_s=0.001),
    )
    await session.run("hello")
    [(attempt, delay, reason)] = watcher.retries
    assert attempt == 1
    # with_retry draws the delay from [0.5, 1.0] of the policy's ceiling: here the
    # configured millisecond, not the default policy's second.
    assert 0.0005 <= delay <= 0.001
    assert reason == "overloaded"


async def test_the_configured_retry_policy_decides_how_many_attempts_are_made(tmp_repo):
    watcher = Watcher()
    script = [ModelError("overloaded", retryable=True, status=529)] * 5
    session = build_session(
        tmp_repo,
        script,
        ui=watcher,
        retry=RetryPolicy(overload_attempts=3, base_delay_s=0.0, max_delay_s=0.0),
    )
    with pytest.raises(ModelError, match="overloaded"):
        await session.run("hello")
    assert len(session.model.requests) == 3  # the default policy would have made five
    assert [attempt for attempt, _, _ in watcher.retries] == [1, 2]


async def test_a_failure_that_is_not_retryable_is_not_retried_and_not_reported(tmp_repo):
    watcher = Watcher()
    session = build_session(
        tmp_repo, [ModelError("bad key", retryable=False, status=401), says("never")], ui=watcher
    )
    with pytest.raises(ModelError, match="bad key"):
        await session.run("hello")
    assert watcher.retries == [] and len(session.model.requests) == 1


async def test_cancelling_while_waiting_to_retry_stops_the_run_at_once(tmp_repo):
    # An hour's back-off: only cancellation can end this in the time the test has.
    retrying = asyncio.Event()

    class NotifyOnRetry(AutoApprove):
        def on_retry(self, _attempt: int, _delay_s: float, _reason: str) -> None:
            retrying.set()

    session = build_session(
        tmp_repo,
        [ModelError("overloaded", retryable=True, status=529), says("never reached")],
        ui=NotifyOnRetry(),
        retry=RetryPolicy(base_delay_s=3600.0, max_delay_s=3600.0),
    )
    task = asyncio.create_task(session.run("hello"))
    await reached(retrying)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert len(session.model.requests) == 1


async def test_the_compaction_request_is_retried_under_the_same_policy(tmp_repo):
    watcher = Watcher()
    session = build_session(
        tmp_repo,
        [says("answer")],
        compact_script=[ModelError("overloaded", retryable=True, status=529), says("SUMMARY")],
        compact_soft=0.00005,
        compact_hard=0.0001,
        ui=watcher,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    assert [attempt for attempt, _, _ in watcher.retries] == [1]
    assert session.compact_model is not None and len(session.compact_model.requests) == 2


# --------------------------------------------------------------------------
# Carried note: context_usage() is async and uses the router's capabilities
# --------------------------------------------------------------------------


async def test_the_context_figure_is_measured_against_the_configured_window(tmp_repo, monkeypatch):
    measured: list[int] = []
    real_verdict = Budget.verdict

    def spying_verdict(self: Budget, transcript: Transcript, system: str, tools: str) -> Any:
        measured.append(self.available())
        return real_verdict(self, transcript, system, tools)

    monkeypatch.setattr(Budget, "verdict", spying_verdict)
    session = build_session(tmp_repo, [says("done")], context_window=64_000)
    await session.run("hello")  # every turn checks the budget before it asks

    _, available = await session.context_usage()
    # 64,000 window, a quarter of it kept for the reply (16,000, which the
    # override bounds the output to), and a tenth held back: 64,000 - 16,000 - 6,400.
    assert available == 41_600
    # And that is the budget the session compacted against, not a second opinion.
    assert measured and set(measured) == {41_600}


async def test_the_context_figure_uses_the_models_table_row_when_nothing_overrides_it(tmp_repo):
    session = build_session(tmp_repo, [])
    _, available = await session.context_usage()
    assert available == available_in(SONNET)


async def test_the_context_figure_is_the_main_roles_when_another_role_has_a_smaller_window(
    tmp_repo,
):
    # The conversation being measured is the main role's. The compact role's model
    # has a window of its own, which is not what the conversation will be sent to.
    assert available_in(HAIKU) != available_in(SONNET)
    session = build_session(tmp_repo, [], compact_script=[])
    _, available = await session.context_usage()
    assert available == available_in(SONNET)


async def test_the_context_figure_sees_capabilities_the_table_does_not_have(tmp_repo):
    # A model the table has never heard of, whose capabilities came from the cache
    # or a probe: the table alone says 8,192 tokens, and would show 5,324.
    session = build_session(
        tmp_repo,
        [],
        capabilities=Capabilities(True, False, "none", context_window=32_000, max_output=4_000),
    )
    _, available = await session.context_usage()
    assert available == 24_800  # 32,000 - 4,000 - 3,200


async def test_the_context_figure_counts_the_conversation_on_top_of_the_fixed_prompt(tmp_repo):
    session = build_session(tmp_repo, [])
    empty, _ = await session.context_usage()
    # What every request carries before the conversation does: at least the
    # system prompt, and the tool definitions beside it.
    assert empty >= HeuristicCounter().count(SYSTEM_PROMPT)

    session.state = replace(session.state, transcript=Transcript((user_text("x" * 3_500),)))
    full, _ = await session.context_usage()
    assert full - empty == 1_000  # 3,500 characters at 3.5 characters a token


async def test_the_context_figure_counts_the_tool_definitions_a_request_carries(tmp_repo):
    session = build_session(tmp_repo, [says("ok")])
    await session.run("hello")
    request = session.model.requests[0]
    session.state = replace(session.state, transcript=Transcript())
    fixed, _ = await session.context_usage()
    counter = HeuristicCounter()
    # The system prompt and, in some form, at least the descriptions of the tools.
    descriptions = "".join(spec.description for spec in request.tools)
    assert fixed >= counter.count(request.system) + counter.count(descriptions)


@pytest.mark.parametrize("capabilities", [None, TEXT_ONLY], ids=["native-tools", "text-protocol"])
async def test_the_budget_is_measured_on_the_system_prompt_that_is_actually_sent(
    tmp_repo, monkeypatch, capabilities
):
    # Under the text protocol the tool definitions are inside the system prompt.
    # A budget measured on the prompt without them would count a request smaller
    # than the one sent, on exactly the small-window models that need the check.
    measured: list[tuple[str, str]] = []
    real_verdict = Budget.verdict

    def spying_verdict(self: Budget, transcript: Transcript, system: str, tools: str) -> Any:
        measured.append((system, tools))
        return real_verdict(self, transcript, system, tools)

    monkeypatch.setattr(Budget, "verdict", spying_verdict)
    session = build_session(tmp_repo, [says("done")], capabilities=capabilities)
    await session.run("hello")
    sent = session.model.requests[0]
    assert measured and {system for system, _ in measured} == {sent.system}
    # The tool definitions a request carries are counted too, in whatever form: at
    # least their descriptions. Under the text protocol none are sent as tools.
    counter = HeuristicCounter()
    descriptions = counter.count("".join(spec.description for spec in sent.tools))
    assert all(counter.count(tools) >= descriptions for _, tools in measured)
    assert (descriptions > 0) == (capabilities is None)


async def test_the_context_is_assembled_once_for_each_request_and_shared_with_the_budget(
    tmp_repo, monkeypatch
):
    # Assembling reads the git state and walks the tree. Done once, the request and
    # the budget cannot see two different prompts; done twice, they could.
    from nanoclaude.context.assemble import assemble as real_assemble

    assembled: list[str] = []

    def counting_assemble(*args: Any, **kwargs: Any) -> Any:
        assembled.append(args[0])
        return real_assemble(*args, **kwargs)

    monkeypatch.setattr("nanoclaude.agent.session.assemble", counting_assemble)
    (tmp_repo / "a.py").write_text("x\n")
    session = build_session(tmp_repo, [calls("Read", {"path": "a.py"}, call_id="t1"), says("done")])
    await session.run("go")
    assert len(session.model.requests) == 2
    assert len(assembled) == 2  # one for each request, and none besides


async def test_a_window_smaller_than_the_fixed_prompt_cannot_be_compacted_and_says_so(tmp_repo):
    session = build_session(tmp_repo, [says("never asked")], context_window=2_000)
    with pytest.raises(ContextTooSmallError, match="Compacting cannot help"):
        await session.run("hello")
    assert session.model.requests == []  # refused before anything was sent


# --------------------------------------------------------------------------
# Carried note: no request sets a temperature
# --------------------------------------------------------------------------


async def test_no_request_the_session_makes_sets_a_temperature(tmp_repo):
    session = build_session(
        tmp_repo,
        [says("SUMMARY"), says("answer")],
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    # The turn really did compact, so the summary request is among those checked.
    assert len(session.model.requests) == 2
    assert sum(is_summary_request(r) for r in session.model.requests) == 1
    assert [r.temperature for r in session.model.requests] == [None, None]


# --------------------------------------------------------------------------
# Compaction: which strategy, from the configured thresholds
# --------------------------------------------------------------------------


async def test_a_conversation_under_the_soft_threshold_is_sent_as_it_is(tmp_repo):
    session = build_session(tmp_repo, [says("answer")])
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    assert len(session.model.requests) == 1  # no summary asked for
    sent = tool_results(session.model.requests[0].transcript)
    assert all("x" * 400 in r.content for r in sent)


@pytest.mark.parametrize(
    ("rounds", "compacts"), [(5, True), (2, False)], ids=["outgrows-the-window", "fits-easily"]
)
async def test_a_conversation_is_compacted_at_the_shipped_thresholds_when_it_fills_a_small_window(
    tmp_repo, rounds, compacts
):
    # No threshold is configured here: it is the window that is small. A model the
    # table does not know, with a 20,000 token window and 2,000 kept for its reply.
    small = Capabilities(True, True, "none", context_window=20_000, max_output=2_000)
    session = build_session(tmp_repo, [says("SUMMARY"), says("answer")], capabilities=small)
    session.state = replace(session.state, transcript=tool_history(rounds, result_chars=10_000))

    used, available = await session.context_usage()
    assert available == available_in(small)
    limits = LimitsConfig()
    if compacts:
        assert used / available >= limits.compact_hard  # what /status shows is over the line
    else:
        assert used / available < limits.compact_soft  # and here it is well under it

    await session.follow_up("what next")
    assert any(is_summary_request(r) for r in session.model.requests) is compacts


async def test_crossing_the_soft_threshold_shrinks_old_tool_results_and_asks_nobody(tmp_repo):
    session = build_session(tmp_repo, [says("answer")], compact_soft=0.0001, compact_hard=0.9999)
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    assert len(session.model.requests) == 1  # micro-compaction needs no model
    sent = session.model.requests[0].transcript
    shrunk = [r.content.startswith("[compacted:") for r in tool_results(sent)]
    # Five rounds, the last three turns kept verbatim: the oldest three shrink.
    assert shrunk == [True, True, True, False, False]
    validate(sent)  # every call still has its answer
    assert len(sent.messages) == 13  # shrunk in place: nothing added, nothing dropped


async def test_crossing_the_hard_threshold_replaces_old_history_with_a_summary(tmp_repo):
    session = build_session(
        tmp_repo,
        [says("SUMMARY of the work", input_tokens=7, output_tokens=3), says("answer")],
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    done = await session.follow_up("what next")

    summary_request, main_request = session.model.requests
    assert is_summary_request(summary_request)
    asked = transcript_text(summary_request.transcript)
    # What was summarised is the head: the first three rounds, not the recent ones.
    assert "result 2" in asked and "result 3" not in asked

    sent = main_request.transcript.messages
    assert sent[0].text().startswith("[earlier conversation, compacted]\nSUMMARY of the work")
    # The summary, then the three most recent turns (six messages) verbatim,
    # ending on the request the person just made.
    assert len(sent) == 7
    assert sent[-1].text() == "what next"
    assert "result 4" in tool_results(main_request.transcript)[-1].content
    assert done.text == "answer"
    # The summary call is billed to the role that made it.
    assert session.router.by_role()["compact"].usage == Usage(7, 3)
    assert session.router.by_role()["main"].usage == Usage()


async def test_the_summary_is_asked_of_the_compact_roles_model_not_the_main_roles(tmp_repo):
    session = build_session(
        tmp_repo,
        [says("answer")],
        compact_script=[says("SUMMARY", input_tokens=5, output_tokens=2)],
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    assert session.compact_model is not None
    assert [is_summary_request(r) for r in session.compact_model.requests] == [True]
    assert not any(is_summary_request(r) for r in session.model.requests)
    # The request that follows is sized by the main model's output limit, not by
    # the compact model's.
    assert HAIKU.max_output != SONNET.max_output
    assert [r.max_output_tokens for r in session.model.requests] == [SONNET.max_output]
    # Priced and attributed at the model that made the call.
    compact = session.router.by_role()["compact"]
    assert (compact.adapter, compact.model) == ("anthropic", "claude-haiku-4-5")
    assert compact.usage == Usage(5, 2)


@pytest.mark.parametrize(
    ("capabilities", "expected"),
    [(None, 2_048), (Capabilities(True, False, "none", 100_000, 1_000), 1_000)],
    ids=["a-large-output-is-capped-at-2048", "a-small-output-is-the-models-own-limit"],
)
async def test_a_summary_asks_for_at_most_2048_tokens_and_never_more_than_the_model_allows(
    tmp_repo, capabilities, expected
):
    session = build_session(
        tmp_repo,
        [says("SUMMARY"), says("answer")],
        capabilities=capabilities,
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    summary_request = session.model.requests[0]
    assert is_summary_request(summary_request)
    assert summary_request.max_output_tokens == expected


async def test_a_summary_is_limited_by_the_compact_models_output_not_the_main_models(tmp_repo):
    # Roles on different models is the point of the router, and their limits differ.
    session = build_session(
        tmp_repo,
        [says("answer")],
        compact_script=[says("SUMMARY")],
        compact_model=UNPRICED,
        compact_capabilities=Capabilities(True, False, "none", 100_000, max_output=1_000),
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    assert session.compact_model is not None
    [summary_request] = session.compact_model.requests
    assert summary_request.max_output_tokens == 1_000  # the main model would allow 2,048


async def test_only_the_text_of_a_summary_is_used_not_the_models_thinking(tmp_repo):
    thought = ModelReply(
        (ThinkingBlock("private reasoning"), TextBlock("THE SUMMARY")),
        StopKind.END_TURN,
        Usage(),
        "scripted",
    )
    session = build_session(
        tmp_repo, [thought, says("answer")], compact_soft=0.00005, compact_hard=0.0001
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.follow_up("what next")
    summary = session.model.requests[1].transcript.messages[0].text()
    assert "THE SUMMARY" in summary and "private reasoning" not in summary


async def test_a_summary_that_fails_stops_the_turn_and_leaves_the_history_alone(tmp_repo):
    session = build_session(
        tmp_repo,
        [says("never asked")],
        compact_script=[ModelError("no route to host", retryable=False)],
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    history = tool_history(5, result_chars=400)
    session.state = replace(session.state, transcript=history)
    with pytest.raises(ModelError) as caught:
        await session.follow_up("what next")
    # The person is told which role's model failed, and why.
    assert "compact" in str(caught.value) and "no route to host" in str(caught.value)
    # Nothing was thrown away for a summary that never arrived: every message the
    # conversation had is still there, and nothing was sent to the main model.
    assert session.state.transcript.messages[: len(history.messages)] == history.messages
    assert session.model.requests == []


async def test_a_summary_that_comes_back_empty_is_refused_and_the_history_kept(tmp_repo):
    session = build_session(
        tmp_repo,
        [says("never asked")],
        compact_script=[says("   \n")],
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    history = tool_history(5, result_chars=400)
    session.state = replace(session.state, transcript=history)
    with pytest.raises(ModelError, match="empty summary"):
        await session.follow_up("what next")
    assert session.state.transcript.messages[: len(history.messages)] == history.messages


async def test_a_manual_compaction_summarises_with_the_instructions_given(tmp_repo):
    session = build_session(tmp_repo, [says("SUMMARY about auth")])
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.compact("the auth module")
    [request] = session.model.requests
    assert is_summary_request(request)
    assert "Pay particular attention to: the auth module" in transcript_text(request.transcript)
    assert (
        session.state.transcript.messages[0]
        .text()
        .startswith("[earlier conversation, compacted]\nSUMMARY about auth")
    )


async def test_a_manual_compaction_without_instructions_adds_none(tmp_repo):
    session = build_session(tmp_repo, [says("SUMMARY")])
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.compact()
    assert "Pay particular attention" not in transcript_text(session.model.requests[0].transcript)


async def test_compacting_a_conversation_with_nothing_older_than_the_recent_turns_asks_nobody(
    tmp_repo,
):
    session = build_session(tmp_repo, [says("never asked")])
    session.state = replace(session.state, transcript=tool_history(1, result_chars=50))
    before = session.state.transcript
    await session.compact()
    assert session.model.requests == [] and session.state.transcript == before


# --------------------------------------------------------------------------
# Persistence: the store mirrors the transcript the model is shown
# --------------------------------------------------------------------------


async def test_after_a_full_compaction_the_store_holds_the_compacted_transcript_and_no_tail(
    tmp_repo,
):
    session = build_session(
        tmp_repo,
        [says("SUMMARY"), says("answer")],
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    done = await session.follow_up("what next")
    assert session.store is not None
    reloaded = session.store.load_transcript(session.session_id)
    # The compacted transcript is shorter than what was stored before it: rows
    # beyond its end must be gone, or a reload would append the old tail.
    assert reloaded == done.state.transcript
    assert reloaded.messages[0].text().startswith("[earlier conversation, compacted]")


async def test_after_a_manual_compaction_the_store_holds_the_compacted_transcript(tmp_repo):
    session = build_session(tmp_repo, [says("SUMMARY")])
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    await session.compact()
    # The next turn would persist it, but the next turn may never come: an
    # explicit /compact is saved when it is done.
    assert session.store is not None
    assert session.store.load_transcript(session.session_id) == session.state.transcript


async def test_after_micro_compaction_the_store_holds_what_the_model_was_shown(tmp_repo):
    session = build_session(tmp_repo, [says("answer")], compact_soft=0.0001, compact_hard=0.9999)
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    done = await session.follow_up("what next")
    assert session.store is not None
    reloaded = session.store.load_transcript(session.session_id)
    assert reloaded == done.state.transcript
    assert reloaded.messages[2].tool_results()[0].content.startswith("[compacted:")


async def test_a_conversation_replaced_from_outside_is_stored_as_the_new_one(tmp_repo):
    # What /clear does: swap the transcript for an empty one and carry on.
    session = build_session(tmp_repo, [says("first answer"), says("second answer")])
    await session.run("first question")
    session.state = replace(session.state, transcript=Transcript(), turn=0)
    done = await session.follow_up("second question")
    assert session.store is not None
    reloaded = session.store.load_transcript(session.session_id)
    assert [m.text() for m in reloaded.messages] == ["second question", "second answer"]
    assert reloaded == done.state.transcript


async def test_each_message_is_written_once_and_a_plain_conversation_is_never_rewritten(
    tmp_repo, monkeypatch
):
    appended: list[int] = []
    rewritten: list[int] = []
    real_append, real_replace = Store.append_message, Store.replace_transcript

    def counting_append(self: Store, session_id: str, seq: int, message: Message) -> None:
        appended.append(seq)
        real_append(self, session_id, seq, message)

    def counting_replace(self: Store, session_id: str, transcript: Transcript) -> None:
        rewritten.append(len(transcript.messages))
        real_replace(self, session_id, transcript)

    monkeypatch.setattr(Store, "append_message", counting_append)
    monkeypatch.setattr(Store, "replace_transcript", counting_replace)
    (tmp_repo / "a.py").write_text("x\n")
    session = build_session(
        tmp_repo, [calls("Read", {"path": "a.py"}, call_id="t1"), says("one"), says("two")]
    )
    await session.run("a")
    await session.follow_up("b")
    assert rewritten == []
    assert appended == list(range(len(session.state.transcript.messages)))  # 0..n-1, once each


async def test_the_call_is_stored_before_the_tool_runs(tmp_repo, monkeypatch):
    # A crash inside a tool must leave the transcript showing what was attempted.
    from nanoclaude.tools.read import ReadTool

    seen_in_store: list[Transcript] = []

    async def look(
        self: ReadTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        assert session.store is not None
        seen_in_store.append(session.store.load_transcript(session.session_id))
        return ToolOutcome(call_id, "seen", is_error=False)

    monkeypatch.setattr(ReadTool, "run", look)
    (tmp_repo / "a.py").write_text("x\n")
    session = build_session(tmp_repo, [calls("Read", {"path": "a.py"}, call_id="t1"), says("done")])
    await session.run("read it")
    [stored] = seen_in_store
    assert [m.role for m in stored.messages] == ["user", "assistant"]
    assert [b.name for b in stored.messages[1].tool_uses()] == ["Read"]


async def test_the_store_holds_what_the_model_is_shown_at_the_moment_it_is_shown(
    tmp_repo, monkeypatch
):
    # Not only once the turn is over: a crash during the request must leave a store
    # that matches what was sent, including when compaction has just replaced the
    # head of the conversation.
    session = build_session(
        tmp_repo,
        [says("SUMMARY"), says("answer")],
        compact_soft=0.00005,
        compact_hard=0.0001,
    )
    session.state = replace(session.state, transcript=tool_history(5, result_chars=400))
    comparisons: list[tuple[Transcript, Transcript]] = []
    real_complete = session.model.complete

    async def spying_complete(request: ModelRequest) -> ModelReply:
        if not is_summary_request(request):
            assert session.store is not None
            stored = session.store.load_transcript(session.session_id)
            comparisons.append((stored, request.transcript))
        return await real_complete(request)

    with monkeypatch.context() as local:
        local.setattr(session.model, "complete", spying_complete)
        await session.follow_up("what next")
    [(stored, sent)] = comparisons
    assert sent.messages[0].text().startswith("[earlier conversation, compacted]")
    assert stored == sent


async def test_at_every_request_the_store_already_holds_what_is_being_sent(tmp_repo, monkeypatch):
    # The person's prompt before the first request, and the tool results before the
    # second: a crash while the model is thinking must not lose either.
    (tmp_repo / "a.py").write_text("x = 1\n")
    session = build_session(
        tmp_repo, [calls("Read", {"path": "a.py"}, call_id="t1"), says("it sets x")]
    )
    comparisons: list[tuple[Transcript, Transcript]] = []
    real_complete = session.model.complete

    async def spying_complete(request: ModelRequest) -> ModelReply:
        assert session.store is not None
        stored = session.store.load_transcript(session.session_id)
        comparisons.append((stored, request.transcript))
        return await real_complete(request)

    with monkeypatch.context() as local:
        local.setattr(session.model, "complete", spying_complete)
        await session.run("what is in a.py")
    assert [len(sent.messages) for _, sent in comparisons] == [1, 3]
    assert all(stored == sent for stored, sent in comparisons)


async def test_a_session_is_created_in_the_store_with_its_directory_and_roles(tmp_repo):
    session = build_session(tmp_repo, [])
    row = stored_row(session)
    assert row["cwd"] == str(tmp_repo)
    assert row["ended_at"] is None and row["total_cost_usd"] is None  # not finished: not known
    assert json.loads(row["roles_json"]) == {
        "main": "m",
        "explore": "m",
        "plan": "m",
        "verify": "m",
        "compact": "m",
        "title": "m",
    }


async def test_a_session_needs_no_database(tmp_repo):
    built = build_session(tmp_repo, [says("done")])
    bare = Session(
        root=built.root,
        config=built.config,
        router=built.router,
        registry=built.registry,
        policy=built.policy,
    )
    assert bare.store is None and bare.audit is None
    assert isinstance(bare.ui, AutoDecline)
    assert len(bare.session_id) == 12
    done = await bare.run("hello")
    assert done.text == "done"
    await bare.aclose()
    assert bare.home == str(Path.home())


# --------------------------------------------------------------------------
# A session survives being interrupted
# --------------------------------------------------------------------------


async def test_cancelling_a_turn_during_a_tool_leaves_a_session_that_can_go_on(
    tmp_repo, monkeypatch
):
    from nanoclaude.tools.write import WriteTool

    started = asyncio.Event()

    async def hang(
        self: WriteTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("the tool was allowed to finish")

    monkeypatch.setattr(WriteTool, "run", hang)
    session = build_session(
        tmp_repo,
        [calls("Write", {"path": "w.txt", "content": "w"}, call_id="w1"), says("picked up again")],
    )
    task = asyncio.create_task(session.run("write it"))
    await reached(started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    done = await session.follow_up("try again")
    assert done.text == "picked up again"
    validate(done.state.transcript)
    # The call that never reported back is answered, honestly: it may have run.
    [result] = tool_results(done.state.transcript)
    assert result.tool_use_id == "w1" and result.is_error
    assert "interrupted" in result.content.lower() and "may or may not" in result.content
    # The model's next request carried that, then the person's new message.
    sent = session.model.requests[1].transcript
    validate(sent)
    assert sent.messages[-1].text() == "try again"
    assert session.store is not None
    assert session.store.load_transcript(session.session_id) == done.state.transcript


async def test_cancelling_a_turn_while_the_model_is_answering_leaves_a_session_that_can_go_on(
    tmp_repo, monkeypatch
):
    started = asyncio.Event()
    session = build_session(tmp_repo, [says("an answer, this time")])

    async def never_answers(request: ModelRequest) -> ModelReply:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("the model was allowed to answer")

    with monkeypatch.context() as local:
        local.setattr(session.model, "complete", never_answers)
        task = asyncio.create_task(session.run("first question"))
        await reached(started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    done = await session.follow_up("second question")
    assert done.text == "an answer, this time"
    validate(done.state.transcript)
    sent = session.model.requests[0].transcript
    assert [m.role for m in sent.messages] == ["user", "assistant", "user"]
    assert sent.messages[0].text() == "first question"
    assert "interrupted" in sent.messages[1].text().lower()
    assert sent.messages[2].text() == "second question"


async def test_a_provider_error_midway_leaves_a_session_that_can_go_on(tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    session = build_session(
        tmp_repo,
        [
            calls("Read", {"path": "a.py"}, call_id="t1"),
            ModelError("bad gateway", retryable=False, status=502),
            says("recovered"),
        ],
    )
    with pytest.raises(ModelError, match="bad gateway"):
        await session.run("read it")
    done = await session.follow_up("try again")
    assert done.text == "recovered"
    validate(done.state.transcript)
    # The tool round that finished before the failure is still in the history.
    assert [r.tool_use_id for r in tool_results(done.state.transcript)] == ["t1"]


async def test_compacting_right_after_an_interrupted_tool_does_not_trip_on_the_open_call(
    tmp_repo, monkeypatch
):
    from nanoclaude.tools.write import WriteTool

    started = asyncio.Event()

    async def hang(
        self: WriteTool, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("the tool was allowed to finish")

    monkeypatch.setattr(WriteTool, "run", hang)
    session = build_session(
        tmp_repo, [calls("Write", {"path": "w.txt", "content": "w"}, call_id="w1")]
    )
    task = asyncio.create_task(session.run("write it"))
    await reached(started)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await session.compact()  # would raise "cannot compact while tool calls are unanswered"
    assert not session.state.transcript.pending_tool_uses()


async def test_each_prompt_gets_the_whole_turn_limit(tmp_repo):
    # Two turns allowed: one tool round and then the answer. Each follow_up uses both.
    (tmp_repo / "a.py").write_text("x\n")
    script = [
        calls("Read", {"path": "a.py"}, call_id="t1"),
        says("first done"),
        calls("Read", {"path": "a.py"}, call_id="t2"),
        says("second done"),
    ]
    session = build_session(tmp_repo, script, max_turns=2)
    first = await session.follow_up("one")
    second = await session.follow_up("two")
    assert (first.reason, first.text) == (StopReason.COMPLETED, "first done")
    assert (second.reason, second.text) == (StopReason.COMPLETED, "second done")
    assert second.state.turn == 1  # counted for this prompt alone


async def test_one_prompt_that_needs_more_turns_than_the_limit_stops_at_it(tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    script = [calls("Read", {"path": "a.py"}, call_id=f"t{i}") for i in range(1, 4)]
    session = build_session(tmp_repo, [*script, says("never reached")], max_turns=2)
    done = await session.follow_up("keep reading")
    assert done.reason is StopReason.TURN_LIMIT
    assert len(session.model.requests) == 2  # the second reply's calls were not run
    validate(done.state.transcript)  # closed cleanly: every call has its answer


async def test_a_prompt_that_hit_the_limit_does_not_use_up_the_next_one(tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    script = [
        calls("Read", {"path": "a.py"}, call_id="t1"),
        calls("Read", {"path": "a.py"}, call_id="t2"),  # the second reply with calls: the limit
        calls("Read", {"path": "a.py"}, call_id="t3"),
        says("recovered"),
    ]
    session = build_session(tmp_repo, script, max_turns=2)
    stopped = await session.follow_up("too much")
    assert stopped.reason is StopReason.TURN_LIMIT
    again = await session.follow_up("try something smaller")
    assert (again.reason, again.text) == (StopReason.COMPLETED, "recovered")


async def test_a_follow_up_before_anything_was_said_starts_the_conversation(tmp_repo):
    # After /clear the REPL calls follow_up on an empty transcript, and the turn
    # limit in force is the configured one, not whatever a placeholder carried.
    (tmp_repo / "a.py").write_text("x\n")
    script = [calls("Read", {"path": "a.py"}, call_id="t1"), says("done")]
    session = build_session(tmp_repo, script, max_turns=5)
    assert session.state.transcript == Transcript() and session.state.max_turns == 5
    done = await session.follow_up("read it")
    assert done.reason is StopReason.COMPLETED and done.text == "done"
    validate(done.state.transcript)


# --------------------------------------------------------------------------
# Wiring: what the session hands to the modules it owns
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["run", "follow_up"])
async def test_a_mention_in_the_prompt_is_inlined_before_the_model_sees_it(tmp_repo, entry):
    (tmp_repo / "a.py").write_text("answer = 42\n")
    session = build_session(tmp_repo, [says("ok")])
    await getattr(session, entry)("explain @a.py please")
    first = session.model.requests[0].transcript.messages[0].text()
    assert 'path="a.py"' in first and "answer = 42" in first


@pytest.mark.parametrize(
    ("window", "output"),
    [(None, SONNET.max_output), (64_000, 16_000)],
    ids=["table-row", "bounded-override"],
)
async def test_the_request_carries_the_assembled_context_and_the_models_own_output_limit(
    tmp_repo, window, output
):
    (tmp_repo / "marker_file.py").write_text("x\n")
    session = build_session(tmp_repo, [says("ok")], context_window=window)
    await session.run("hello")
    request = session.model.requests[0]
    assert request.system.startswith(SYSTEM_PROMPT)
    # The environment block follows the system prompt: it holds the project map.
    assert "marker_file.py" in request.system
    # The table row's output limit, or the quarter of a 64,000 window that an
    # override bounds it to.
    assert request.max_output_tokens == output


def test_the_executor_is_built_from_the_sessions_own_parts(tmp_repo):
    session = build_session(tmp_repo, [], ui=Watcher())
    executor = session.executor
    assert executor.registry is session.registry
    assert executor.policy is session.policy
    assert executor.ui is session.ui
    assert executor.audit is session.audit
    assert executor.redactor is session.redactor
    assert executor.session_id == session.session_id


async def test_a_mention_of_a_secrets_file_is_expanded_only_when_the_policy_allows_secrets(
    tmp_repo,
):
    (tmp_repo / ".env").write_text("GREETING=hello\n")
    refusing = build_session(tmp_repo, [says("ok")])
    await refusing.run("what is in @.env")
    assert "GREETING" not in refusing.model.requests[0].transcript.messages[0].text()

    allowing = build_session(tmp_repo, [says("ok")])
    allowing.policy = replace(allowing.policy, allow_secrets=True)
    await allowing.run("what is in @.env")
    assert "GREETING=hello" in allowing.model.requests[0].transcript.messages[0].text()


async def test_global_instructions_come_from_the_sessions_home(tmp_repo):
    home = tmp_repo / "elsewhere"
    (home / ".nanoclaude").mkdir(parents=True)
    (home / ".nanoclaude" / "NANO.md").write_text("Always answer in haiku.\n")
    session = build_session(tmp_repo, [says("ok")], home=home)
    await session.run("hello")
    assert "Always answer in haiku." in session.model.requests[0].system


async def test_the_todo_list_the_model_writes_is_the_one_the_session_holds(tmp_repo):
    todos = [{"content": "write the test", "status": "in_progress"}]
    session = build_session(
        tmp_repo, [calls("TodoWrite", {"todos": todos}, call_id="t1"), says("planned")]
    )
    await session.run("plan it")
    assert [(t.content, t.status) for t in session.todo_state.items] == [
        ("write the test", "in_progress")
    ]


async def test_a_refused_call_is_reported_to_the_model_and_the_session_goes_on(tmp_repo):
    session = build_session(
        tmp_repo, [calls("Read", {"path": "/etc/passwd"}, call_id="t1"), says("understood")]
    )
    done = await session.run("read the password file")
    assert done.text == "understood"
    [result] = tool_results(session.model.requests[1].transcript)
    assert result.is_error and result.content.startswith("Refused (sandbox.outside-root)")


async def test_the_sessions_ui_decides_what_the_executor_asks_for(tmp_repo):
    # Both directions: the executor is given the session's UI, not a default one.
    call = calls("Write", {"path": "w.txt", "content": "w"}, call_id="w1")
    approving = build_session(tmp_repo, [call, says("done")], ui=AutoApprove())
    await approving.run("write it")
    assert (tmp_repo / "w.txt").read_text() == "w"

    (tmp_repo / "w.txt").unlink()
    declining = build_session(
        tmp_repo,
        [calls("Write", {"path": "w.txt", "content": "w"}, call_id="w2"), says("done")],
        ui=AutoDecline(),
    )
    done = await declining.run("write it")
    assert not (tmp_repo / "w.txt").exists()
    [result] = tool_results(done.state.transcript)
    assert result.is_error and "declined" in result.content


async def test_a_call_is_billed_to_the_model_the_router_used_whatever_the_sessions_config_says(
    tmp_repo,
):
    # The router decides which client answers a role, so it decides which model the
    # call is billed at. A front end that swaps the session's own config (as /model
    # does) has not changed the client, and must not change the bill.
    session = build_session(tmp_repo, [says("done", input_tokens=10)])
    session.config = replace(
        session.config,
        models={"x": ModelConfig("anthropic", "claude-opus-5")},
        roles=RolesConfig("x", "x", "x", "x", "x", "x"),
    )
    await session.run("hello")
    main = session.router.by_role()["main"]
    assert (main.adapter, main.model) == ("anthropic", "claude-sonnet-5")


async def test_every_reply_reaches_the_ui_once(tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    watcher = Watcher()
    session = build_session(
        tmp_repo, [calls("Read", {"path": "a.py"}, call_id="t1"), says("done")], ui=watcher
    )
    await session.run("go")
    assert [r.stop for r in watcher.replies] == [StopKind.TOOL_USE, StopKind.END_TURN]


async def test_the_audit_log_is_kept_under_the_sessions_own_id(tmp_repo):
    (tmp_repo / "a.py").write_text("x\n")
    session = build_session(
        tmp_repo,
        [
            calls_many(("Read", {"path": "a.py"}, "t1"), ("Read", {"path": "a.py"}, "t2")),
            says("done"),
        ],
    )
    await session.run("go")
    assert session.store is not None
    rows = session.store.db.execute(
        "SELECT tool_use_id, session_id, outcome FROM tool_calls ORDER BY tool_use_id"
    ).fetchall()
    assert [(r["tool_use_id"], r["session_id"], r["outcome"]) for r in rows] == [
        ("t1", session.session_id, "ok"),
        ("t2", session.session_id, "ok"),
    ]


# --------------------------------------------------------------------------
# Closing
# --------------------------------------------------------------------------


async def test_closing_a_session_closes_its_clients_and_its_store(tmp_repo):
    session = build_session(tmp_repo, [says("done")])
    await session.run("hello")
    closed_before = session.model.closed  # a variable: mypy would narrow the attribute itself
    await session.aclose()
    assert closed_before is False and session.model.closed
    assert session.store is not None
    with pytest.raises(RuntimeError, match="store is not open"):
        _ = session.store.db


async def test_the_store_is_closed_even_when_a_client_fails_to_close(tmp_repo, monkeypatch):
    session = build_session(tmp_repo, [])

    async def refuse() -> None:
        raise RuntimeError("would not close")

    monkeypatch.setattr(session.router, "aclose", refuse)
    with pytest.raises(RuntimeError, match="would not close"):
        await session.aclose()
    assert session.store is not None
    with pytest.raises(RuntimeError, match="store is not open"):
        _ = session.store.db
