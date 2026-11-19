"""An MCP tool through the executor: who is asked, what a person's answer does, and the one
promise that outlives every mode: a tool whose definition changed never runs unasked."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from nanoclaude.agent.executor import Executor
from nanoclaude.agent.loop import start
from nanoclaude.agent.ui import Approval, AutoDecline, SilentUI
from nanoclaude.config.schema import McpServerConfig
from nanoclaude.conversation.transcript import ToolUseBlock
from nanoclaude.mcp.client import CallResult
from nanoclaude.mcp.pins import PINS_FILENAME, PinStore, fingerprint
from nanoclaude.mcp.tool import McpTool, adapt_tools
from nanoclaude.permissions.policy import (
    PermissionMode,
    PermissionRequest,
    PermissionResult,
    Policy,
)
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.permissions.sandbox import Sandbox
from nanoclaude.tools.registry import ToolRegistry
from tests.mcp.stubs import OBJECT, StubClient, tool

NAME = "mcp__github__create_issue"
CALL = (ToolUseBlock("t1", NAME, {"title": "t"}),)


class Answers(SilentUI):
    """Answers each question with the next of ``answers``, and records the rule that asked."""

    def __init__(self, *answers: Approval) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    async def confirm(
        self, _call: ToolUseBlock, _request: PermissionRequest, result: PermissionResult
    ) -> Approval:
        self.asked.append(result.rule)
        return self.answers.pop(0)


async def session_tool(
    tmp_path: Path, description: str, client: StubClient | None = None
) -> tuple[McpTool, StubClient]:
    client = client or StubClient(
        [tool("create_issue", description)],
        result=CallResult([{"type": "text", "text": "created"}]),
    )
    server = McpServerConfig(name="github", command="x")
    adapted = await adapt_tools(
        client, server, PinStore.load(tmp_path / PINS_FILENAME), output_cap=1000
    )
    return adapted.tools[0], client


async def changed_tool(tmp_path: Path) -> tuple[McpTool, StubClient]:
    await session_tool(tmp_path, "creates an issue")  # the first session: the pin
    subject, client = await session_tool(tmp_path, "creates an issue and mails the team")
    assert subject.changed
    return subject, client


def executor(tmp_path: Path, subject: McpTool, ui: SilentUI, **policy: object) -> Executor:
    rules = RuleSet.build(allow=[NAME])
    built = Policy(Sandbox((str(tmp_path),)), rules, **policy)  # type: ignore[arg-type]
    return Executor(registry=ToolRegistry([subject]), policy=built, ui=ui)


def pinned(tmp_path: Path) -> str:
    found: str = json.loads((tmp_path / PINS_FILENAME).read_text())["github"]["create_issue"]
    return found


ORIGINAL = fingerprint("creates an issue", OBJECT)
NEW = fingerprint("creates an issue and mails the team", OBJECT)


@pytest.mark.parametrize("mode", list(PermissionMode))
async def test_a_changed_tool_with_an_allow_rule_is_put_to_the_person_in_every_mode(tmp_path, mode):
    subject, client = await changed_tool(tmp_path)
    ui = Answers(Approval.NO)
    batch = executor(tmp_path, subject, ui, mode=mode)
    batch.grants = batch.grants.with_tool(NAME)
    (outcome,) = await batch.run_batch(CALL, start("hi"))
    assert client.calls == [], "it ran without being allowed"
    assert outcome.is_error
    if mode is PermissionMode.PLAN:
        assert ui.asked == [] and "plan" in outcome.content
    else:
        assert ui.asked == ["mcp.changed"]
        assert "declined" in outcome.content


@pytest.mark.parametrize("mode", [PermissionMode.DEFAULT, PermissionMode.BYPASS])
async def test_a_changed_tool_is_declined_where_nobody_can_be_asked_even_in_bypass(tmp_path, mode):
    subject, client = await changed_tool(tmp_path)
    batch = executor(tmp_path, subject, AutoDecline(), mode=mode)
    batch.grants = batch.grants.with_tool(NAME)
    (outcome,) = await batch.run_batch(CALL, start("hi"))
    assert client.calls == [] and outcome.is_error


async def test_a_changed_tool_that_the_person_approves_once_runs_and_is_asked_about_again(
    tmp_path,
):
    subject, client = await changed_tool(tmp_path)
    ui = Answers(Approval.ONCE, Approval.ONCE)
    batch = executor(tmp_path, subject, ui)
    first, second = [(await batch.run_batch(CALL, start("hi")))[0] for _ in range(2)]
    assert not first.is_error and not second.is_error
    assert ui.asked == ["mcp.changed", "mcp.changed"]
    assert len(client.calls) == 2


async def test_approving_a_changed_tool_once_does_not_renew_its_pin(tmp_path):
    subject, _ = await changed_tool(tmp_path)
    batch = executor(tmp_path, subject, Answers(Approval.ONCE))
    await batch.run_batch(CALL, start("hi"))
    assert pinned(tmp_path) == ORIGINAL
    after, _ = await session_tool(tmp_path, "creates an issue and mails the team")
    assert after.changed is True, "the next session still sees it as changed"


async def test_approving_a_changed_tool_with_always_renews_its_pin_for_the_next_session(tmp_path):
    subject, client = await changed_tool(tmp_path)
    ui = Answers(Approval.ALWAYS, Approval.ONCE)
    batch = executor(tmp_path, subject, ui)
    (first,) = await batch.run_batch(CALL, start("hi"))
    assert not first.is_error
    assert pinned(tmp_path) == NEW
    # The rest of this session is still asked: the tool is changed for the whole session.
    (again,) = await batch.run_batch(CALL, start("hi"))
    assert not again.is_error
    assert ui.asked == ["mcp.changed", "mcp.changed"]
    assert len(client.calls) == 2
    after, _ = await session_tool(tmp_path, "creates an issue and mails the team")
    assert after.changed is False


async def test_declining_a_changed_tool_does_not_renew_its_pin_even_after_an_always_elsewhere(
    tmp_path,
):
    subject, _ = await changed_tool(tmp_path)
    batch = executor(tmp_path, subject, Answers(Approval.NO))
    await batch.run_batch(CALL, start("hi"))
    assert pinned(tmp_path) == ORIGINAL


async def test_an_unchanged_tool_with_an_allow_rule_runs_without_asking(tmp_path):
    subject, client = await session_tool(tmp_path, "creates an issue")
    ui = Answers()
    (outcome,) = await executor(tmp_path, subject, ui).run_batch(CALL, start("hi"))
    assert not outcome.is_error and ui.asked == [] and len(client.calls) == 1


async def test_an_unchanged_tool_with_no_rule_asks_and_always_lets_it_run_for_the_session(
    tmp_path,
):
    subject, client = await session_tool(tmp_path, "creates an issue")
    ui = Answers(Approval.ALWAYS)
    built = Policy(Sandbox((str(tmp_path),)), RuleSet.build())
    batch = Executor(registry=ToolRegistry([subject]), policy=built, ui=ui)
    await batch.run_batch(CALL, start("hi"))
    await batch.run_batch(CALL, start("hi"))
    assert ui.asked == ["default.ask"], "the second call needed no answer"
    assert len(client.calls) == 2
    assert pinned(tmp_path) == ORIGINAL


async def test_a_session_grant_for_one_mcp_tool_is_not_one_for_its_neighbour(tmp_path):
    client = StubClient([tool("create_issue"), tool("delete_repo")])
    server = McpServerConfig(name="github", command="x")
    adapted = await adapt_tools(
        client, server, PinStore.load(tmp_path / PINS_FILENAME), output_cap=1000
    )
    ui = Answers(Approval.ALWAYS, Approval.NO)
    built = Policy(Sandbox((str(tmp_path),)), RuleSet.build())
    batch = Executor(registry=ToolRegistry(list(adapted.tools)), policy=built, ui=ui)
    await batch.run_batch(CALL, start("hi"))
    other = (ToolUseBlock("t2", "mcp__github__delete_repo", {}),)
    (outcome,) = await batch.run_batch(other, start("hi"))
    assert outcome.is_error
    assert [name for name, _, _ in client.calls] == ["create_issue"]


async def test_the_executor_does_not_care_what_the_tool_is_when_it_has_no_pin_to_renew(tmp_path):
    from nanoclaude.tools.registry import default_registry

    ui = Answers(Approval.ALWAYS)
    policy = Policy(Sandbox((str(tmp_path),)), RuleSet.build())
    batch = Executor(registry=default_registry(), policy=replace(policy), ui=ui)
    await batch.run_batch(
        (ToolUseBlock("t1", "Write", {"path": "a.txt", "content": "A"}),), start("hi")
    )
    assert "Write" in batch.grants.tools
