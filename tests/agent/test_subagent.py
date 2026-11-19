"""Tests for nanoclaude.agent.subagent: the ``Task`` tool and the ``explore`` sub-agent.

A sub-agent is a session nested in the one that called it, so these run whole sessions from
scripted models: the parent's script calls ``Task``, and the ``explore`` role's own script is
what the sub-agent's model says. Each invariant has a test that fails when its guard is taken
out. Nothing sleeps: the wall-clock limit is tried with a model that never answers and a limit
of a few hundredths of a second, so what the test waits for is the guard and not the clock.
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, NoReturn

import pytest

from nanoclaude.agent import subagent
from nanoclaude.agent.router import Router
from nanoclaude.agent.session import Session
from nanoclaude.agent.subagent import (
    DEFAULT_LIMITS,
    RunGate,
    SubAgentLimits,
    build_registry,
    run_subagent,
)
from nanoclaude.agent.ui import Approval, AutoApprove
from nanoclaude.conversation.transcript import (
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Transcript,
    assistant_text,
    user_text,
)
from nanoclaude.permissions.policy import (
    ALLOWED_WITHOUT_ASKING,
    READ_ONLY_TOOLS,
    Decision,
    PermissionMode,
    PermissionRequest,
    PermissionResult,
    evaluate,
)
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import WiderSandboxError
from nanoclaude.providers.base import (
    ModelError,
    ModelReply,
    ModelRequest,
    StopKind,
    ToolSpec,
    Usage,
)
from nanoclaude.providers.capabilities import Capabilities
from nanoclaude.providers.texttools import MAX_PARSE_RETRIES
from nanoclaude.subagents import EXPLORE_PROMPT, SUBAGENTS, SubAgent, SubAgentCall
from nanoclaude.testing.scripted import calls, says
from nanoclaude.testing.session import ScriptedClient, ScriptedSession, build_session
from nanoclaude.tools.base import ToolContext, ToolOutcome, cap_bytes, ok
from nanoclaude.tools.registry import ToolRegistry
from nanoclaude.tools.task import TaskTool
from nanoclaude.tools.write import WriteTool
from tests.keys import pem

#: The model the explore role is routed to in these tests: not the main one, and priced, so that
#: a call made on it is told apart from one made on the main model by its cost.
EXPLORE_MODEL = "claude-haiku-4-5"

TASK_ID = "task_1"


@pytest.fixture(autouse=True)
def _no_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """The main session's environment block asks git for the branch: not what is tested here."""
    monkeypatch.setattr("nanoclaude.context.assemble.git_state", lambda _root: "")
    monkeypatch.setattr("nanoclaude.context.instructions.run_git", _git_is_not_asked)


def _git_is_not_asked(*_args: object, **_kwargs: object) -> NoReturn:
    raise OSError("git is not run by these tests")


class Watcher(AutoApprove):
    """Approves what it is asked, and remembers what it was asked and told."""

    def __init__(self) -> None:
        self.confirmed: list[str] = []
        self.replies: list[ModelReply] = []
        self.requests: list[str] = []
        self.texts: list[str] = []
        self.retries: list[tuple[int, float, str]] = []

    async def confirm(
        self, call: ToolUseBlock, request: PermissionRequest, result: PermissionResult
    ) -> Approval:
        self.confirmed.append(call.name)
        return await super().confirm(call, request, result)

    def on_reply(self, reply: ModelReply) -> None:
        self.replies.append(reply)

    def on_request_start(self, role: str) -> None:
        self.requests.append(role)

    def on_text(self, delta: str) -> None:
        self.texts.append(delta)

    def on_retry(self, attempt: int, delay_s: float, reason: str) -> None:
        self.retries.append((attempt, delay_s, reason))


def ask_explore(prompt: str = "where is it defined?", *, call_id: str = TASK_ID) -> ModelReply:
    return calls("Task", {"agent": "explore", "prompt": prompt}, call_id=call_id)


def build(
    root: Path,
    explore: list[ModelReply | ModelError],
    *,
    parent: list[ModelReply | ModelError] | None = None,
    ui: AutoApprove | None = None,
    **options: Any,
) -> ScriptedSession:
    """A session whose model asks the explore sub-agent one question and then finishes."""
    script = parent if parent is not None else [ask_explore(), says("All done.")]
    options.setdefault("explore_model", EXPLORE_MODEL)
    return build_session(root, script, explore_script=explore, ui=ui, **options)


def tool_names(request: ModelRequest) -> set[str]:
    return {spec.name for spec in request.tools}


def results_in(transcript: Transcript) -> list[ToolResultBlock]:
    return [
        block
        for message in transcript.messages
        for block in message.blocks
        if isinstance(block, ToolResultBlock)
    ]


def tool_result_for(session: ScriptedSession, call_id: str = TASK_ID) -> ToolResultBlock:
    found = [r for r in results_in(session.state.transcript) if r.tool_use_id == call_id]
    assert len(found) == 1, f"{len(found)} results for {call_id} in {session.state.transcript}"
    return found[0]


def explore_results(session: ScriptedSession) -> list[ToolResultBlock]:
    """What the explore model was shown as tool results, in its last request."""
    assert session.explore_model is not None
    return results_in(session.explore_model.requests[-1].transcript)


def with_limits(monkeypatch: pytest.MonkeyPatch, **limits: Any) -> None:
    monkeypatch.setattr(subagent, "DEFAULT_LIMITS", replace(DEFAULT_LIMITS, **limits))


# --------------------------------------------------------------------------
# What a Task call does, and whose model does it
# --------------------------------------------------------------------------


async def test_the_sub_agent_is_answered_by_the_explore_roles_model_and_not_the_main_one(tmp_path):
    session = build(tmp_path, [says("It is defined in a.py:3.")])
    await session.run("find it")

    assert session.explore_model is not None
    assert len(session.explore_model.requests) == 1
    # The main model was asked twice (the Task call, and the answer after its result) and was
    # never handed the sub-agent's prompt as a request of its own.
    assert len(session.model.requests) == 2
    first = session.explore_model.requests[0].transcript.messages[0]
    assert first == user_text("where is it defined?")
    assert tool_result_for(session).content == "It is defined in a.py:3."
    assert not tool_result_for(session).is_error


async def test_the_sub_agents_cost_is_recorded_under_the_explore_role(tmp_path):
    session = build(
        tmp_path,
        [
            calls("Glob", {"pattern": "*.py"}, call_id="sub_1", input_tokens=700, output_tokens=40),
            says("Found.", input_tokens=900, output_tokens=60),
        ],
        parent=[
            calls("Task", {"agent": "explore", "prompt": "q"}, call_id=TASK_ID, input_tokens=100),
            says("Done.", input_tokens=200, output_tokens=10),
        ],
    )
    await session.run("go")

    by_role = session.router.by_role()
    assert list(by_role) == ["main", "explore"], "/cost lists the roles that were used"
    assert by_role["explore"].usage == Usage(1600, 100)
    assert by_role["explore"].model == EXPLORE_MODEL
    assert by_role["main"].usage == Usage(300, 10), "the sub-agent's tokens are not the main's"
    # Priced at the model that made the calls, as the router prices every call.
    expected = session.router.prices.cost(Usage(1600, 100), "anthropic", EXPLORE_MODEL)
    assert expected is not None and by_role["explore"].cost == pytest.approx(expected)
    assert session.cost == pytest.approx(
        (by_role["main"].cost or 0.0) + (by_role["explore"].cost or 0.0)
    )


async def test_the_parents_transcript_has_one_result_for_the_call_and_none_of_the_sub_agents_turns(
    tmp_path,
):
    session = build(
        tmp_path,
        [
            calls("Glob", {"pattern": "*.py"}, call_id="sub_1", preamble="looking at the files"),
            says("The answer."),
        ],
    )
    await session.run("find it")

    messages = session.state.transcript.messages
    assert [m.role for m in messages] == ["user", "assistant", "user", "assistant"], messages
    assert [b.tool_use_id for b in results_in(session.state.transcript)] == [TASK_ID]
    uses = [b.name for m in messages for b in m.blocks if isinstance(b, ToolUseBlock)]
    assert uses == ["Task"], "no call of the sub-agent's is in the parent's transcript"
    said = [b.text for m in messages for b in m.blocks if isinstance(b, TextBlock)]
    assert "looking at the files" not in said
    # And none of it was stored: the store holds the four messages of the parent.
    assert session.store is not None
    stored = session.store.load_transcript(session.session_id)
    assert stored == session.state.transcript
    rows = session.store.db.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
    assert rows == 1, "the sub-agent did not open a session row of its own"


async def test_the_person_is_shown_a_request_in_progress_and_never_the_sub_agents_words(tmp_path):
    ui = Watcher()
    session = build(tmp_path, [says("Hidden words.")], ui=ui)
    await session.run("find it")

    assert ui.requests == ["main", "explore", "main"], "the explore request is shown as one"
    assert [r.blocks for r in ui.replies if r.stop is StopKind.END_TURN] == [
        (TextBlock("All done."),)
    ]
    assert "".join(ui.texts) == "All done."
    assert ui.confirmed == ["Task"], "only the Task call was asked about"


async def test_a_retry_of_the_sub_agents_request_is_shown_to_the_person_and_the_run_goes_on(
    tmp_path,
):
    ui = Watcher()
    overloaded = ModelError("overloaded", retryable=True, status=529)
    session = build(tmp_path, [overloaded, says("Found it.")], ui=ui)
    await session.run("find it")

    assert [attempt for attempt, _, _ in ui.retries] == [1]
    assert tool_result_for(session).content == "Found it."


async def test_what_comes_back_is_scrubbed_and_stripped_like_any_tool_result(tmp_path):
    block, body = pem()
    session = build(tmp_path, [says("\x1b[31mred\x1b[0m\n" + "\n".join(block))])
    await session.run("find it")

    content = tool_result_for(session).content
    assert "\x1b" not in content and content.startswith("[31mred") is False
    assert body[0] not in content and "[redacted:private-key]" in content, content


# --------------------------------------------------------------------------
# What the sub-agent is, and what it is told
# --------------------------------------------------------------------------


async def test_the_sub_agent_has_read_glob_and_grep_and_nothing_else(tmp_path):
    session = build(tmp_path, [says("ok")])
    await session.run("find it")

    assert session.explore_model is not None
    assert tool_names(session.explore_model.requests[0]) == {"Read", "Glob", "Grep"}
    # What the parent was offered is more, which is what makes the difference worth a test.
    assert {"Task", "Write", "Edit", "TodoWrite"} <= tool_names(session.model.requests[0])
    assert [tool.name for tool in build_registry(SUBAGENTS["explore"])] == ["Read", "Glob", "Grep"]


class Probe:
    """A read-only tool that only the parent has. Nothing in a sub-agent's persona names it."""

    name = "Probe"
    read_only = True

    def spec(self) -> ToolSpec:
        return ToolSpec("Probe", "A tool the parent was given.", {"type": "object"})

    def permission_request(
        self, _ctx: ToolContext, _arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        return PermissionRequest("Probe", "probe")

    async def run(
        self, _ctx: ToolContext, call_id: str, _arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        return ok(call_id, "probed")


async def test_the_sub_agents_tools_are_the_ones_its_persona_names_and_not_the_parents_less_some(
    tmp_path,
):
    """A tool added to the parent is not in the sub-agent. A registry made by taking Task and the
    write tools out of the parent's would hold ``Probe``: this one is made from a list of names."""
    session = build(tmp_path, [calls("Probe", {}, call_id="sub_1"), says("ok")])
    session.registry = ToolRegistry([*session.registry, Probe()])
    await session.run("find it")

    assert "Probe" in tool_names(session.model.requests[0]), "the parent was not given it"
    assert session.explore_model is not None
    assert tool_names(session.explore_model.requests[0]) == {"Read", "Glob", "Grep"}
    (result,) = explore_results(session)
    assert result.is_error and "tool.unknown" in result.content, result


async def test_a_sub_agent_that_asks_for_task_is_told_that_there_is_no_such_tool(tmp_path):
    session = build(
        tmp_path,
        [calls("Task", {"agent": "explore", "prompt": "again"}, call_id="sub_1"), says("ok")],
    )
    await session.run("find it")

    (result,) = explore_results(session)
    assert result.is_error and "tool.unknown" in result.content, result
    assert session.explore_model is not None and len(session.explore_model.requests) == 2


async def test_a_sub_agent_that_asks_for_a_write_tool_is_told_that_there_is_no_such_tool(tmp_path):
    session = build(
        tmp_path,
        [calls("Write", {"path": "x.txt", "content": "hi"}, call_id="sub_1"), says("ok")],
    )
    await session.run("find it")

    assert not (tmp_path / "x.txt").exists()
    (result,) = explore_results(session)
    assert result.is_error and "tool.unknown" in result.content, result


async def test_the_sub_agents_system_prompt_is_the_fixed_one_and_not_the_projects(tmp_path):
    (tmp_path / "NANO.md").write_text("Always answer in haiku.\n")
    home = tmp_path / "home"
    session = build_session(
        tmp_path,
        [ask_explore(), says("done")],
        explore_script=[says("ok")],
        explore_model=EXPLORE_MODEL,
        home=home,
    )
    await session.run("find it")

    assert session.explore_model is not None
    system = session.explore_model.requests[0].system
    assert system == f"{EXPLORE_PROMPT}\n\nThe working directory is {session.root}."
    assert "haiku" not in system and "Project structure" not in system
    assert "haiku" in session.model.requests[0].system, "the parent's prompt does have it"
    for phrase in ("read-only", "Read, Glob and Grep", "Answer only what you were asked"):
        assert phrase in EXPLORE_PROMPT, phrase
    assert "concise conclusion" in EXPLORE_PROMPT


async def test_a_sub_agent_on_a_text_protocol_model_is_given_the_tools_in_its_prompt(tmp_path):
    text_only = Capabilities(
        native_tools=False,
        parallel_tools=False,
        cache="none",
        context_window=32_000,
        max_output=4_000,
    )
    session = build(
        tmp_path,
        [says("ok")],
        explore_capabilities=text_only,
        explore_model="a-model-the-table-lacks",
    )
    await session.run("find it")

    assert session.explore_model is not None
    request = session.explore_model.requests[0]
    assert request.tools == (), "the definitions are in the prompt, as the protocol"
    assert request.system.startswith(EXPLORE_PROMPT)
    defined = {line[4:] for line in request.system.splitlines() if line.startswith("### ")}
    assert defined == {"Read", "Glob", "Grep"}, defined


async def test_an_at_sign_in_the_prompt_is_not_a_mention_to_inline(tmp_path):
    (tmp_path / "a.txt").write_text("the secret contents\n")
    session = build(tmp_path, [says("ok")], parent=[ask_explore("look at @a.txt"), says("done")])
    await session.run("find it")

    assert session.explore_model is not None
    assert session.explore_model.requests[0].transcript.messages[0] == user_text("look at @a.txt")


# --------------------------------------------------------------------------
# The sandbox, the rules and the grants
# --------------------------------------------------------------------------


def test_a_session_cannot_be_given_a_role_that_is_not_one_of_the_six(tmp_path):
    parent = build(tmp_path, [says("x")])
    with pytest.raises(ValueError, match="unknown role 'judge'"):
        Session(
            root=parent.root,
            config=parent.config,
            router=parent.router,
            registry=parent.registry,
            policy=parent.policy,
            role="judge",
        )


def parent_call(prompt: str = "q") -> SubAgentCall:
    return SubAgentCall("explore", prompt, TASK_ID)


async def test_a_sub_agents_sandbox_cannot_be_wider_than_its_parents(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    session = build(root, [says("ok")])

    for wider in (str(tmp_path), "/", str(tmp_path / "elsewhere")):
        with pytest.raises(WiderSandboxError):
            await run_subagent(session, parent_call(), audit_turn=0, root=wider)
    assert session.explore_model is not None and session.explore_model.requests == []


async def test_a_sub_agents_sandbox_can_be_narrower_than_its_parents(tmp_path):
    (tmp_path / "inner").mkdir()
    (tmp_path / "inner" / "in.txt").write_text("inside\n")
    (tmp_path / "out.txt").write_text("outside\n")
    session = build(
        tmp_path,
        [
            calls("Read", {"path": "in.txt"}, call_id="sub_1"),
            calls("Read", {"path": str(tmp_path / "out.txt")}, call_id="sub_2"),
            says("ok"),
        ],
        parent=[says("unused")],
    )
    result = await run_subagent(session, parent_call(), audit_turn=0, root=str(tmp_path / "inner"))

    assert not result.is_error
    first, second = explore_results(session)
    assert "inside" in first.content and not first.is_error
    assert second.is_error and "sandbox.outside-root" in second.content, second


async def test_without_a_narrower_root_the_sub_agent_has_the_parents_sandbox(tmp_path):
    (tmp_path / "a.txt").write_text("in the project\n")
    session = build(tmp_path, [calls("Read", {"path": "a.txt"}, call_id="sub_1"), says("ok")])
    await session.run("find it")

    (result,) = explore_results(session)
    assert "in the project" in result.content and not result.is_error


async def test_a_sub_agent_is_held_to_the_parents_deny_rules(tmp_path):
    (tmp_path / "private.txt").write_text("not for the model\n")
    session = build(
        tmp_path,
        [calls("Read", {"path": "private.txt"}, call_id="sub_1"), says("ok")],
    )
    policy = replace(session.policy, rules=RuleSet.build(deny=["Read(private.txt)"]))
    session.policy = policy
    session.executor.policy = policy
    await session.run("find it")

    (result,) = explore_results(session)
    assert result.is_error and "rule.deny" in result.content, result


@pytest.fixture
def scribe(monkeypatch: pytest.MonkeyPatch) -> SubAgent:
    """A sub-agent that is given a tool that asks: the only way to put a question to one.

    Nothing built in has such a tool, since a read tool is never asked about, so the persona
    is made here and Write is made available to it for the length of the test.
    """
    monkeypatch.setitem(subagent.TOOLS, "Write", WriteTool)
    return SubAgent("scribe", "writes", "explore", ("Read", "Write"), "You write files.")


def write_call(path: str = "new.txt") -> ModelReply:
    return calls("Write", {"path": path, "content": "hello\n"}, call_id="sub_1")


async def test_a_question_inside_a_sub_agent_is_declined_and_the_person_is_never_asked(
    tmp_path, scribe
):
    ui = Watcher()
    session = build(tmp_path, [write_call(), says("ok")], parent=[says("unused")], ui=ui)
    result = await run_subagent(session, parent_call(), audit_turn=0, agent=scribe)

    assert ui.confirmed == [], "the person was asked about a call the sub-agent made"
    assert not (tmp_path / "new.txt").exists(), "a declined write was written"
    (declined,) = explore_results(session)
    assert declined.is_error
    assert "Write needs the person's confirmation, and a sub-agent may not ask for it." in (
        declined.content
    ), declined
    assert "declined" not in declined.content, "nobody was asked, and nobody said no"
    assert not result.is_error


async def test_the_sub_agent_cannot_add_to_the_session_grants_and_the_parents_are_unchanged(
    tmp_path, scribe
):
    ui = Watcher()
    session = build(tmp_path, [write_call(), says("ok")], parent=[says("unused")], ui=ui)
    before = session.executor.grants
    await run_subagent(session, parent_call(), audit_turn=0, agent=scribe)

    assert session.executor.grants == before
    assert "Write" not in session.executor.grants.tools


async def test_the_sub_agent_inherits_the_session_grants_the_person_has_given(tmp_path, scribe):
    ui = Watcher()
    session = build(tmp_path, [write_call(), says("ok")], parent=[says("unused")], ui=ui)
    session.executor.grants = session.executor.grants.with_tool("Write")
    await run_subagent(session, parent_call(), audit_turn=0, agent=scribe)

    assert (tmp_path / "new.txt").read_text() == "hello\n", "a grant of the parent's did not hold"
    assert ui.confirmed == []


async def test_a_sub_agent_that_is_given_the_task_tool_still_cannot_start_another(
    tmp_path, monkeypatch
):
    """Nesting is one deep: the tool is not in its registry, and where a persona put it there
    anyway, its executor has nothing to start a sub-agent with."""
    monkeypatch.setitem(subagent.TOOLS, "Task", TaskTool)
    nester = SubAgent("nester", "delegates", "explore", ("Read", "Task"), "You delegate.")
    session = build(
        tmp_path,
        [calls("Task", {"agent": "explore", "prompt": "again"}, call_id="sub_1"), says("ok")],
        parent=[says("unused")],
    )
    # Granted, so that the call is not declined at the question and gets as far as the tool.
    session.executor.grants = session.executor.grants.with_tool("Task")
    await run_subagent(session, parent_call(), audit_turn=0, agent=nester)

    assert session.explore_model is not None
    assert len(session.explore_model.requests) == 2, "a second sub-agent was asked something"
    (result,) = explore_results(session)
    assert result.is_error and "not available" in result.content, result


@pytest.mark.parametrize(
    ("mode", "written", "said"),
    [
        (PermissionMode.ACCEPT_EDITS, True, ""),
        (PermissionMode.PLAN, False, "mode.plan-read-only"),
        (PermissionMode.DEFAULT, False, "may not ask"),
    ],
)
async def test_a_sub_agent_is_judged_in_the_parents_permission_mode(
    tmp_path, scribe, mode, written, said
):
    session = build(tmp_path, [write_call(), says("ok")], parent=[says("unused")])
    session.policy = replace(session.policy, mode=mode)
    session.executor.policy = session.policy
    await run_subagent(session, parent_call(), audit_turn=0, agent=scribe)

    assert (tmp_path / "new.txt").exists() is written
    (outcome,) = explore_results(session)
    assert said in outcome.content, outcome


# --------------------------------------------------------------------------
# The limits
# --------------------------------------------------------------------------


def test_the_limits_are_thirty_calls_two_hundred_thousand_tokens_and_ten_minutes():
    assert SubAgentLimits(30, 200_000, 600.0) == DEFAULT_LIMITS
    assert subagent._duration(600.0) == "10 minutes of wall-clock time"


async def test_a_sub_agent_stopped_before_it_has_said_anything_comes_back_with_only_the_note(
    tmp_path, monkeypatch
):
    with_limits(monkeypatch, max_input_tokens=10)
    session = build(tmp_path, [says("never asked")])
    await session.run("find it")

    assert session.explore_model is not None and session.explore_model.requests == []
    content = tool_result_for(session).content
    assert content == (
        "(explore returned no text)\n\n"
        "[explore stopped: it reached its limit of 10 estimated input tokens]"
    )


def test_a_persona_that_names_a_tool_a_sub_agent_cannot_be_given_is_refused_by_name():
    persona = SubAgent("odd", "does odd things", "explore", ("Read", "Bash"), "You are odd.")
    with pytest.raises(ValueError, match="names Bash, which a sub-agent cannot be given"):
        build_registry(persona)


def steps(count: int) -> list[ModelReply | ModelError]:
    """``count`` replies that each say where they are and ask for one more search."""
    return [
        calls("Glob", {"pattern": "*"}, call_id=f"sub_{n}", preamble=f"step {n}")
        for n in range(1, count + 1)
    ]


async def test_a_sub_agent_is_stopped_after_thirty_model_calls_and_its_last_text_comes_back(
    tmp_path,
):
    session = build(tmp_path, steps(40))
    await session.run("find it")

    assert session.explore_model is not None
    assert len(session.explore_model.requests) == 30, "the 31st request was sent"
    result = tool_result_for(session)
    assert not result.is_error
    assert result.content.startswith("step 30"), result.content
    assert "[explore stopped: it reached its limit of 30 model calls]" in result.content


async def test_a_sub_agent_is_stopped_before_it_goes_past_its_input_token_limit(
    tmp_path, monkeypatch
):
    (tmp_path / "big.txt").write_text("\n".join(f"line {n} " + "x" * 80 for n in range(1500)))
    with_limits(monkeypatch, max_input_tokens=5_000)
    session = build(
        tmp_path,
        [calls("Read", {"path": "big.txt"}, call_id="sub_1", preamble="reading it"), says("late")],
    )
    await session.run("find it")

    assert session.explore_model is not None
    assert len(session.explore_model.requests) == 1, "the request that would not fit was sent"
    result = tool_result_for(session)
    assert result.content.startswith("reading it"), result.content
    assert "its limit of 5,000 estimated input tokens]" in result.content


def test_the_gate_counts_the_estimates_it_lets_through_and_refuses_the_one_that_goes_past():
    gate = RunGate(SubAgentLimits(max_model_calls=10, max_input_tokens=250))
    gate(100)
    gate(100)
    with pytest.raises(subagent.LimitReached, match="250 estimated input tokens"):
        gate(100)
    assert (gate.calls, gate.input_tokens) == (2, 200), "a refused request was counted"
    gate(50)  # exactly the limit is not past it


class Hangs(ScriptedClient):
    """Answers from its script, and when that is spent never answers, and notes it if it is
    cancelled while it waits."""

    cancelled = False

    async def complete(
        self, request: ModelRequest, *, on_text: Callable[[str], None] | None = None
    ) -> ModelReply:
        if self._script:
            return await super().complete(request, on_text=on_text)
        self.requests.append(request)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


async def test_a_sub_agent_whose_model_never_answers_is_stopped_at_the_wall_clock_limit(
    tmp_path, monkeypatch
):
    with_limits(monkeypatch, max_seconds=0.05)
    session = build(tmp_path, [])
    hanging = Hangs([calls("Glob", {"pattern": "*"}, call_id="sub_1", preamble="so far")])
    session.router = Router(
        session.router.config,
        session.router.cache,
        {"m": session.model, "e": hanging},
        prices=session.router.prices,
    )
    # Bounded from outside, so that a stop that is missing is a failure and not a hung suite.
    await asyncio.wait_for(session.run("find it"), timeout=10)

    result = tool_result_for(session)
    assert not result.is_error
    assert result.content.startswith("so far"), result.content
    assert "[explore stopped: it reached its limit of 0.05 seconds of wall-clock time]" in (
        result.content
    )
    assert len(hanging.requests) == 2
    # The request that never answered was cancelled, and not left running behind the result: a
    # limit that stopped waiting for it and let it go on would pass everything above.
    assert hanging.cancelled, "the request in flight was not cancelled"
    left = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    assert left == [], f"tasks left running after the sub-agent was stopped: {left}"


async def test_a_timeout_of_the_providers_own_is_not_taken_for_the_wall_clock_limit(tmp_path):
    class SocketTimesOut(ScriptedClient):
        async def complete(
            self, _request: ModelRequest, *, on_text: Callable[[str], None] | None = None
        ) -> ModelReply:
            del on_text
            raise TimeoutError("the socket timed out")

    session = build(tmp_path, [])
    session.router = Router(
        session.router.config,
        session.router.cache,
        {"m": session.model, "e": SocketTimesOut([])},
        prices=session.router.prices,
    )
    await session.run("find it")

    result = tool_result_for(session)
    assert result.is_error
    assert "TimeoutError: the socket timed out" in result.content
    assert "wall-clock" not in result.content


async def test_the_loops_own_turn_limit_is_not_what_stops_a_sub_agent(tmp_path):
    """The person's ``max_turns`` is for their conversation: with it at 3, a sub-agent that
    takes five turns is not cut short by it."""
    session = build(
        tmp_path,
        [*steps(5), says("finished")],
        max_turns=3,
    )
    await session.run("find it")

    assert tool_result_for(session).content == "finished"


# --------------------------------------------------------------------------
# What comes back
# --------------------------------------------------------------------------


async def test_an_answer_longer_than_the_output_cap_is_cut_with_a_note_of_how_much(tmp_path):
    session = build(tmp_path, [says("a" * 300)])
    session.config = replace(
        session.config, limits=replace(session.config.limits, output_cap_bytes=100)
    )
    await session.run("find it")

    content = tool_result_for(session).content
    body, note = content.split("\n", 1)
    assert body == "a" * 100
    assert note == "[output cut: 200 bytes beyond the 100-byte limit]"


async def test_an_answer_within_the_output_cap_is_returned_whole(tmp_path):
    session = build(tmp_path, [says("a" * 100)])
    session.config = replace(
        session.config, limits=replace(session.config.limits, output_cap_bytes=100)
    )
    await session.run("find it")

    assert tool_result_for(session).content == "a" * 100


def test_the_cut_falls_between_characters_and_counts_every_byte_that_was_dropped():
    cut = cap_bytes("é" * 10, 5)  # two bytes each: a fifth byte is half of a letter
    body, note = cut.split("\n")
    assert body == "éé"
    assert note == "[output cut: 16 bytes beyond the 5-byte limit]"
    assert cap_bytes("abc", 3) == "abc"
    assert cap_bytes("", 0) == ""


async def test_the_cap_is_the_configured_one_and_not_a_figure_of_the_sub_agents(tmp_path):
    session = build(tmp_path, [says("b" * 300)])
    assert session.config.limits.output_cap_bytes == 100_000
    await session.run("find it")

    assert tool_result_for(session).content == "b" * 300


async def test_a_sub_agent_that_says_nothing_is_reported_as_having_said_nothing(tmp_path):
    session = build(
        tmp_path,
        [ModelReply((TextBlock(" "),), StopKind.END_TURN, Usage(), "scripted")],
    )
    await session.run("find it")

    assert tool_result_for(session).content == "(explore returned no text)"


@pytest.mark.parametrize(
    ("stop", "said"),
    [
        (StopKind.MAX_TOKENS, "cut off at the model's output limit"),
        (StopKind.REFUSAL, "declined to continue"),
    ],
)
async def test_a_reply_that_ended_for_another_reason_than_finishing_is_said_to_have(
    tmp_path, stop, said
):
    reply = ModelReply((TextBlock("what there was"),), stop, Usage(), "scripted")
    session = build(tmp_path, [reply])
    await session.run("find it")

    content = tool_result_for(session).content
    assert content.startswith("what there was")
    assert "[explore ended early: " in content and said in content, content
    assert not tool_result_for(session).is_error


async def test_a_sub_agent_whose_model_fails_is_a_tool_error_with_what_it_had_written(tmp_path):
    session = build(
        tmp_path,
        [
            calls("Glob", {"pattern": "*"}, call_id="sub_1", preamble="halfway there"),
            ModelError("the provider is down"),
        ],
    )
    await session.run("find it")

    result = tool_result_for(session)
    assert result.is_error
    assert "The explore sub-agent could not finish: the provider is down." in result.content
    assert "halfway there" in result.content
    # The parent carried on: it was told, and answered.
    assert session.state.transcript.messages[-1] == assistant_text("All done.")


async def test_a_sub_agent_whose_model_cannot_write_a_tool_call_is_told_to_the_parent_as_an_error(
    tmp_path,
):
    text_only = Capabilities(
        native_tools=False,
        parallel_tools=False,
        cache="none",
        context_window=32_000,
        max_output=4_000,
    )
    bad = says('<tool name="Glob">{"pattern": </tool>')
    session = build(
        tmp_path,
        [bad] * (MAX_PARSE_RETRIES + 1),
        explore_capabilities=text_only,
        explore_model="a-model-the-table-lacks",
    )
    await session.run("find it")

    result = tool_result_for(session)
    assert result.is_error and "--role explore=<model>" in result.content, result


# --------------------------------------------------------------------------
# The audit
# --------------------------------------------------------------------------


def audit_rows(session: ScriptedSession) -> list[sqlite3.Row]:
    assert session.store is not None
    return session.store.db.execute(
        "SELECT tool_use_id, turn, tool, parent_call_id, outcome FROM tool_calls ORDER BY ts, rowid"
    ).fetchall()


async def test_the_sub_agents_calls_are_audited_under_the_task_calls_turn_and_id(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    session = build(
        tmp_path,
        [
            calls("Read", {"path": "a.txt"}, call_id="sub_1"),
            calls("Glob", {"pattern": "*"}, call_id="sub_2"),
            says("ok"),
        ],
    )
    await session.run("find it")

    rows = [tuple(row) for row in audit_rows(session)]
    assert rows == [
        (TASK_ID, 0, "Task", None, "ok"),
        ("sub_1", 0, "Read", TASK_ID, "ok"),
        ("sub_2", 0, "Glob", TASK_ID, "ok"),
    ], rows


async def test_a_later_task_call_is_audited_under_its_own_turn(tmp_path):
    session = build(
        tmp_path,
        [
            calls("Glob", {"pattern": "*"}, call_id="sub_1"),
            says("ok"),
            calls("Glob", {"pattern": "*"}, call_id="sub_2"),
            says("ok"),
        ],
        parent=[ask_explore(), says("first done"), ask_explore(call_id="task_2"), says("again")],
    )
    await session.run("one")
    await session.follow_up("two")

    turns = {row["tool_use_id"]: row["turn"] for row in audit_rows(session)}
    assert turns == {TASK_ID: 0, "sub_1": 0, "task_2": 1, "sub_2": 1}, turns


async def test_a_call_the_sub_agent_was_refused_is_audited_with_its_parent_too(tmp_path):
    session = build(
        tmp_path,
        [calls("Read", {"path": "/etc/hostname"}, call_id="sub_1"), says("ok")],
    )
    await session.run("find it")

    refused = [tuple(r) for r in audit_rows(session) if r["tool_use_id"] == "sub_1"]
    assert refused == [("sub_1", 0, "Read", TASK_ID, "refused")]


# --------------------------------------------------------------------------
# Task calls the model gets wrong
# --------------------------------------------------------------------------


async def test_a_task_call_for_an_agent_that_does_not_exist_is_an_error_that_lists_explore(
    tmp_path,
):
    ui = Watcher()
    session = build(
        tmp_path,
        [says("never asked")],
        parent=[calls("Task", {"agent": "plan", "prompt": "q"}, call_id=TASK_ID), says("ok")],
        ui=ui,
    )
    await session.run("find it")

    result = tool_result_for(session)
    assert result.is_error
    assert "no sub-agent named 'plan'" in result.content and "explore" in result.content, result
    assert ui.confirmed == [], "the person was asked about a call that cannot succeed"
    assert session.explore_model is not None and session.explore_model.requests == []
    assert [tuple(r) for r in audit_rows(session)] == [(TASK_ID, 0, "Task", None, "refused")]


async def test_a_task_call_that_the_person_declines_never_starts_the_sub_agent(tmp_path):
    class Declines(Watcher):
        async def confirm(
            self, call: ToolUseBlock, _request: PermissionRequest, _result: PermissionResult
        ) -> Approval:
            self.confirmed.append(call.name)
            return Approval.NO

    ui = Declines()
    session = build(tmp_path, [says("never asked")], ui=ui)
    await session.run("find it")

    assert ui.confirmed == ["Task"]
    result = tool_result_for(session)
    assert result.is_error and result.content.startswith("The user declined to run Task.")
    assert session.explore_model is not None and session.explore_model.requests == []


async def test_a_task_call_is_not_asked_about_when_a_rule_allows_it(tmp_path):
    ui = Watcher()
    session = build(tmp_path, [says("found")], ui=ui)
    session.policy = replace(session.policy, rules=RuleSet.build(allow=["Task"]))
    session.executor.policy = session.policy
    await session.run("find it")

    assert ui.confirmed == []
    assert tool_result_for(session).content == "found"


def test_task_asks_by_default_and_a_rule_can_allow_it(tmp_path):
    session = build(tmp_path, [says("x")])
    request = PermissionRequest("Task", "explore")

    decided = evaluate(request, session.policy)
    assert (decided.decision, decided.rule) == (Decision.ASK, "default.ask")
    allowed = evaluate(request, replace(session.policy, rules=RuleSet.build(allow=["Task"])))
    assert (allowed.decision, allowed.rule) == (Decision.ALLOW, "rule.allow")
    assert "Task" not in READ_ONLY_TOOLS and "Task" not in ALLOWED_WITHOUT_ASKING
