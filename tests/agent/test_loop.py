from collections.abc import Hashable
from typing import Any

import pytest

from nanoclaude.agent.loop import (
    DEFAULT_MAX_TURNS,
    Done,
    LoopError,
    LoopState,
    RunTools,
    StopReason,
    observe,
    resume,
    start,
    step,
)
from nanoclaude.conversation.transcript import Transcript, validate
from nanoclaude.testing.scripted import calls, calls_many, says
from nanoclaude.tools.base import ToolOutcome


class FakeOutcome:
    # Fully annotated, not just given a bare ``-> None``: mypy's
    # disallow_incomplete_defs stays on for tests (only disallow_untyped_defs and
    # disallow_untyped_calls are relaxed there), so a return annotation with
    # still-bare parameters is "incomplete" and fails strict mode -- annotating
    # only the return type would trade the ruff ANN204 finding for a mypy one.
    def __init__(
        self,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
        observed: tuple[tuple[str, Any], ...] = (),
    ) -> None:
        self.tool_use_id, self.content = tool_use_id, content
        self.is_error, self.observed = is_error, observed


def test_a_text_reply_finishes_the_loop():
    outcome = step(start("hi"), says("done"))
    assert isinstance(outcome, Done)
    assert outcome.text == "done"
    assert outcome.reason is StopReason.COMPLETED


def test_a_tool_reply_asks_for_the_calls():
    outcome = step(start("hi"), calls("Read", {"path": "a.py"}, call_id="t1"))
    assert isinstance(outcome, RunTools)
    assert [c.id for c in outcome.calls] == ["t1"]


def test_observe_takes_the_real_tool_outcome_and_not_only_a_stand_in():
    # Observation says ToolOutcome satisfies it, and nothing checked that: every
    # other test here passes FakeOutcome, a class whose fields can be assigned.
    # ToolOutcome is a frozen dataclass, whose cannot, so a protocol of plain
    # attributes is refused by mypy --strict at the one call that matters, the
    # session's. This is that call, typed.
    outcomes: tuple[ToolOutcome, ...] = (ToolOutcome("t1", "1\tx = 1"),)
    outcome = step(start("hi"), calls("Read", {"path": "a.py"}, call_id="t1"))
    state = observe(outcome.state, outcomes)
    assert state.turn == 1
    last = state.transcript.last()
    assert last is not None and last.tool_results()[0].content == "1\tx = 1"


def test_observe_advances_the_turn_and_records_what_was_seen():
    outcome = step(start("hi"), calls("Read", {"path": "a.py"}, call_id="t1"))
    assert isinstance(outcome, RunTools)
    state = observe(outcome.state, [FakeOutcome("t1", "1\tx = 1", observed=(("/p/a.py", "SHA"),))])
    assert state.turn == 1
    assert state.stamp_for("/p/a.py") == "SHA"
    validate(state.transcript)


def test_results_must_answer_every_pending_call_in_order():
    outcome = step(start("hi"), calls_many(("Read", {}, "t1"), ("Glob", {}, "t2")))
    assert isinstance(outcome, RunTools)
    with pytest.raises(LoopError, match=r"expected results for \['t1', 't2'\]"):
        observe(outcome.state, [FakeOutcome("t2", "b"), FakeOutcome("t1", "a")])


def test_turn_limit_closes_the_transcript_rather_than_leaving_calls_unanswered():
    state = start("hi", max_turns=1)
    outcome = step(state, calls("Bash", {"command": "ls"}, call_id="t1"))
    assert isinstance(outcome, Done)
    assert outcome.reason is StopReason.TURN_LIMIT
    validate(outcome.state.transcript)
    refusals = outcome.state.transcript.messages[-2]
    # The key assertions: validate() alone would not catch a left-unanswered
    # tool_use -- a trailing pending call is legal mid-turn transcript shape
    # (that's what makes a normal RunTools state valid too), so it passes
    # whether or not the loop actually closed this call out. What proves it
    # did is that the message after the call carries a matching, error
    # tool_result naming the turn limit.
    assert refusals.tool_results()[0].is_error
    assert "turn limit" in refusals.tool_results()[0].content


def test_a_turn_limit_stop_ends_with_text_so_no_turn_is_left_looking_interrupted():
    # Spec 5.4: the unanswered calls get a "not executed" result, "then ends with text".
    # A transcript that ended on those results would be a user message nobody answered,
    # which is what an interrupted turn looks like, and the next prompt would be told
    # that this one was interrupted.
    outcome = step(start("hi", max_turns=1), calls("Bash", {"command": "ls"}, call_id="t1"))
    assert isinstance(outcome, Done) and outcome.reason is StopReason.TURN_LIMIT
    last = outcome.state.transcript.messages[-1]
    assert last.role == "assistant"
    assert last.text() == outcome.text
    assert [m.role for m in outcome.state.transcript.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_the_turn_limit_text_says_how_many_turns_were_taken():
    state = start("hi", max_turns=3)
    for call_id in ("t1", "t2"):
        outcome = step(state, calls("Read", {"path": "a.py"}, call_id=call_id))
        assert isinstance(outcome, RunTools)
        state = observe(outcome.state, [FakeOutcome(call_id, "x")])
    final = step(state, calls("Read", {"path": "a.py"}, call_id="t3"))
    assert isinstance(final, Done) and final.reason is StopReason.TURN_LIMIT
    assert final.text == "Stopped after 3 turns without finishing the task."
    assert final.state.turn == 3


def test_a_conversation_that_stopped_at_the_turn_limit_can_be_continued():
    outcome = step(start("hi", max_turns=1), calls("Bash", {"command": "ls"}, call_id="t1"))
    assert isinstance(outcome, Done)
    resumed = resume(outcome.state, "try again")
    validate(resumed.transcript)
    assert resumed.transcript.messages[-1].text() == "try again"


def test_refusal_is_distinguished_from_completion():
    from nanoclaude.conversation.transcript import TextBlock
    from nanoclaude.providers.base import ModelReply, StopKind, Usage

    reply = ModelReply((TextBlock("no"),), StopKind.REFUSAL, Usage(), "scripted")
    outcome = step(start("hi"), reply)
    assert isinstance(outcome, Done)
    assert outcome.reason is StopReason.REFUSAL


def test_usage_accumulates_across_turns():
    # A single-turn version of this test cannot tell "accumulates" from
    # "replaces with the latest reply's usage" apart -- state.usage starts at
    # Usage() (all zeros), so state.usage + reply.usage and plain reply.usage
    # are bit-for-bit identical on turn one. Spanning two non-zero-usage turns
    # is what makes a dropped "+" show up as a wrong number instead of an
    # accidental match.
    from nanoclaude.providers.base import Usage

    first_reply = calls("Read", {"path": "a.py"}, call_id="t1", input_tokens=10, output_tokens=3)
    outcome = step(start("hi"), first_reply)
    assert isinstance(outcome, RunTools)
    assert outcome.state.usage == Usage(10, 3)

    observed = observe(outcome.state, [FakeOutcome("t1", "1\tx = 1")])
    final = step(observed, says("done", input_tokens=7, output_tokens=2))
    assert isinstance(final, Done)
    assert final.state.usage == Usage(17, 5)


def test_read_state_survives_a_second_observe():
    # A regression that rebuilds read_state from scratch each call (``{}``
    # instead of ``dict(state.read_state)``) would still pass a test with only
    # one observe() -- there would be nothing earlier for it to have lost.
    first = step(start("hi"), calls("Read", {"path": "a.py"}, call_id="t1"))
    assert isinstance(first, RunTools)
    after_first = observe(
        first.state, [FakeOutcome("t1", "1\tx = 1", observed=(("/p/a.py", "SHA1"),))]
    )

    second = step(after_first, calls("Read", {"path": "b.py"}, call_id="t2"))
    assert isinstance(second, RunTools)
    after_second = observe(
        second.state, [FakeOutcome("t2", "1\ty = 2", observed=(("/p/b.py", "SHA2"),))]
    )

    assert after_second.stamp_for("/p/a.py") == "SHA1"
    assert after_second.stamp_for("/p/b.py") == "SHA2"


def test_an_empty_reply_is_a_loop_error():
    from nanoclaude.providers.base import ModelReply, StopKind, Usage

    with pytest.raises(LoopError, match="empty reply"):
        step(start("hi"), ModelReply((), StopKind.END_TURN, Usage(), "scripted"))


# The tests below close gaps the brief's own 8 tests left open: resume(), the
# max_turns guard, observe()'s "nothing pending" guard, and turns_left all had
# no test at all (line-coverage showed 93%, missing exactly these branches).
# The brief's REFUSAL/MAX_TOKENS test was also split in two below, one case
# per name (it was originally named for both but its body only ever
# constructed a REFUSAL reply) -- the reason mapping is a dict literal, not a
# branch, so statement coverage stays green regardless and would not have
# surfaced the missing MAX_TOKENS case on its own.


def test_max_tokens_is_distinguished_from_completion():
    from nanoclaude.conversation.transcript import TextBlock
    from nanoclaude.providers.base import ModelReply, StopKind, Usage

    reply = ModelReply((TextBlock("truncated"),), StopKind.MAX_TOKENS, Usage(), "scripted")
    outcome = step(start("hi"), reply)
    assert isinstance(outcome, Done)
    assert outcome.reason is StopReason.MAX_TOKENS


def test_max_turns_must_be_at_least_one():
    with pytest.raises(LoopError, match="max_turns must be at least 1, got 0"):
        start("hi", max_turns=0)


def test_resume_appends_a_user_message_when_nothing_is_pending():
    outcome = step(start("hi"), says("done"))
    assert isinstance(outcome, Done)
    resumed = resume(outcome.state, "and another thing")
    assert len(resumed.transcript) == len(outcome.state.transcript) + 1
    last = resumed.transcript.last()
    assert last is not None
    assert last.role == "user"
    assert last.text() == "and another thing"


def _a_conversation_one_tool_round_in() -> LoopState:
    """A conversation that has run one tool round and answered, with something read and used."""
    first = step(
        start("hi", max_turns=3), calls("Read", {"path": "a.py"}, call_id="t1", input_tokens=7)
    )
    assert isinstance(first, RunTools)
    observed = (("/p/a.py", "SHA"),)
    answered = observe(first.state, [FakeOutcome("t1", "1\tx = 1", observed=observed)])
    done = step(answered, says("done", input_tokens=5, output_tokens=2))
    assert isinstance(done, Done) and done.state.turn == 1
    return done.state


def test_a_new_prompt_starts_the_turn_count_again():
    # max_turns bounds what one prompt may take, not what a whole conversation may.
    state = _a_conversation_one_tool_round_in()
    resumed = resume(state, "and another thing")
    assert resumed.turn == 0
    assert resumed.turns_left == resumed.max_turns == 3


def test_a_new_prompt_keeps_what_the_conversation_has_read_and_used():
    # Starting the count again must not start the conversation again: an Edit still
    # needs the file to have been read, and the usage is the conversation's.
    state = _a_conversation_one_tool_round_in()
    resumed = resume(state, "and another thing")
    assert resumed.read_state == {"/p/a.py": "SHA"}
    assert resumed.usage == state.usage and resumed.usage.input_tokens == 12
    assert resumed.transcript.messages[: len(state.transcript)] == state.transcript.messages


def test_resume_refuses_while_tool_calls_are_unanswered():
    outcome = step(start("hi"), calls("Read", {"path": "a.py"}, call_id="t1"))
    assert isinstance(outcome, RunTools)
    with pytest.raises(LoopError, match="cannot resume while tool calls are unanswered"):
        resume(outcome.state, "and another thing")


def test_observe_refuses_when_nothing_is_pending():
    with pytest.raises(LoopError, match="no tool calls outstanding"):
        observe(start("hi"), [])


def test_turns_left_counts_down_as_turns_are_used():
    outcome = step(start("hi", max_turns=3), calls("Read", {"path": "a.py"}, call_id="t1"))
    assert isinstance(outcome, RunTools)
    assert outcome.state.turns_left == 3
    state = observe(outcome.state, [FakeOutcome("t1", "1\tx = 1")])
    assert state.turns_left == 2


def test_turns_left_never_goes_negative():
    # Unreachable through step()/observe() -- the loop always stops at or
    # before turn == max_turns -- so this is checked by constructing the
    # otherwise-impossible state directly, the same way tests/conversation
    # builds Transcript values the smart constructors would never produce.
    state = LoopState(Transcript(()), turn=5, max_turns=3)
    assert state.turns_left == 0


def test_stamp_for_is_none_when_nothing_was_observed_at_that_path():
    assert start("hi").stamp_for("/never/seen.py") is None


# Review round 1 closed five more findings. Three are below as their own
# tests; the other two (extending test_usage_accumulates_across_turns to a
# real second turn, and splitting the old combined REFUSAL/MAX_TOKENS test)
# are above, next to the tests they changed.
#
# The first three: LoopState (and Done/RunTools, which embed it) had the exact
# defect Task 2 fixed on ToolUseBlock -- frozen=True with the default eq=True
# generates a real __hash__, so isinstance(x, Hashable) says True while
# hash(x) actually raises, naming "mappingproxy" (from read_state) rather than
# the class. Pinned the same way Task 2 pins ToolUseBlock: both directions,
# so a fix that makes one true without the other is still caught.


def test_loop_state_is_not_hashable():
    state = start("hi")
    assert not isinstance(state, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: 'LoopState'"):
        hash(state)


def test_done_is_not_hashable():
    outcome = step(start("hi"), says("done"))
    assert isinstance(outcome, Done)
    assert not isinstance(outcome, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: 'Done'"):
        hash(outcome)


def test_run_tools_is_not_hashable():
    outcome = step(start("hi"), calls("Read", {"path": "a.py"}, call_id="t1"))
    assert isinstance(outcome, RunTools)
    assert not isinstance(outcome, Hashable)  # type: ignore[unreachable]
    with pytest.raises(TypeError, match="unhashable type: 'RunTools'"):
        hash(outcome)


def test_start_defaults_max_turns_to_the_module_constant():
    # A silent change to DEFAULT_MAX_TURNS passes every other test in this
    # file -- none of them depend on its actual value, only on max_turns
    # eventually being reached.
    assert start("hi").max_turns == DEFAULT_MAX_TURNS


def test_turn_limit_closes_every_pending_call_not_just_the_first():
    # The single-call version above cannot tell "closes every pending call"
    # from "closes the first one" apart. validate() would catch a dropped or
    # misordered refusal on its own (transcript.py's own pairing rule), so
    # this is about a friendly, specific failure rather than silent
    # corruption -- but without it, closing only calls[:1] would surface as an
    # unhandled TranscriptError deep in step(), not a named assertion here.
    state = start("hi", max_turns=1)
    outcome = step(state, calls_many(("Read", {}, "t1"), ("Glob", {}, "t2")))
    assert isinstance(outcome, Done)
    assert outcome.reason is StopReason.TURN_LIMIT
    validate(outcome.state.transcript)
    results = outcome.state.transcript.messages[-2].tool_results()
    assert [r.tool_use_id for r in results] == ["t1", "t2"]
    assert all(r.is_error for r in results)
    assert all("turn limit" in r.content for r in results)
