"""Tests for nanoclaude.tools.task: what the model is told, what a call is judged by, and what a
call that cannot run says. The sub-agent that a call starts is tested in tests/agent."""

from __future__ import annotations

from dataclasses import replace

import pytest

from nanoclaude.permissions.policy import Decision, evaluate
from nanoclaude.permissions.rules import RuleSet
from nanoclaude.subagents import SUBAGENTS, SubAgentCall, SubAgentResult, subagent_names
from nanoclaude.tools.base import Tool, ToolArgumentError, ToolContext
from nanoclaude.tools.registry import default_registry
from nanoclaude.tools.task import DESCRIPTION, SCHEMA, TaskTool
from tests.keys import pem

#: The description as spec 17.10 gives it. A second copy on purpose: it is a prompt, and a change
#: to it is a change to what the model is told, which needs a design note (CONTRIBUTING.md).
SPEC_DESCRIPTION = (
    "Delegate a self-contained piece of work to a sub-agent.\n"
    "\n"
    "The sub-agent starts with an empty conversation, runs its own tool calls, and returns\n"
    "only its final answer. Use it when the work would otherwise fill your context with\n"
    "detail you do not need to keep — searching a large codebase for where something is\n"
    "defined, for example.\n"
    "\n"
    "- `agent` selects which sub-agent to use; the user can list them with /agents.\n"
    "- Give it a complete, self-contained prompt. It cannot see this conversation and cannot\n"
    "  ask you follow-up questions.\n"
    "- It cannot reach outside the working directory, and it cannot be granted permissions\n"
    "  you do not have."
)


class Spawned:
    """A spawner that records what it was asked and answers as told."""

    def __init__(self, result: SubAgentResult) -> None:
        self.result = result
        self.calls: list[SubAgentCall] = []

    async def __call__(self, call: SubAgentCall) -> SubAgentResult:
        self.calls.append(call)
        return self.result


def test_the_tool_is_registered_and_satisfies_the_tool_protocol():
    registry = default_registry()
    assert "Task" in registry
    assert isinstance(registry.get("Task"), Tool)


def test_the_description_is_the_one_in_the_spec_word_for_word():
    assert DESCRIPTION == SPEC_DESCRIPTION
    assert TaskTool().spec().description == SPEC_DESCRIPTION


def test_the_schema_has_an_agent_and_a_prompt_that_are_both_strings_and_both_required():
    spec = TaskTool().spec()
    assert spec.name == "Task"
    assert dict(spec.schema) == dict(SCHEMA)
    assert SCHEMA["required"] == ["agent", "prompt"]
    assert {key: value["type"] for key, value in SCHEMA["properties"].items()} == {
        "agent": "string",
        "prompt": "string",
    }
    assert SCHEMA["additionalProperties"] is False


def test_a_task_call_is_not_read_only():
    assert TaskTool().read_only is False


def test_a_call_is_judged_by_the_agent_it_names_and_touches_no_path(ctx):
    request = TaskTool().permission_request(ctx, {"agent": "explore", "prompt": "where is x?"})
    assert (request.tool, request.subject) == ("Task", "explore")
    assert request.resolved_paths == () and request.is_write is False


def test_a_rule_can_name_the_agent_a_task_call_is_for(ctx):
    request = TaskTool().permission_request(ctx, {"agent": "explore", "prompt": "q"})
    allow = replace(ctx.policy, rules=RuleSet.build(allow=["Task(explore)"]))
    deny = replace(ctx.policy, rules=RuleSet.build(deny=["Task(explore)"]))
    assert evaluate(request, allow).decision is Decision.ALLOW
    assert evaluate(request, deny).decision is Decision.DENY


@pytest.mark.parametrize("agent", ["plan", "general", "Explore", " explore", "", "explore/../plan"])
def test_an_agent_that_does_not_exist_is_refused_with_the_names_there_are(ctx, agent):
    with pytest.raises(ToolArgumentError) as caught:
        TaskTool().permission_request(ctx, {"agent": agent, "prompt": "q"})
    message = str(caught.value)
    assert f"no sub-agent named {agent!r}" in message
    assert "available sub-agents are: explore" in message


def test_the_names_a_call_is_told_are_the_sub_agents_there_are():
    assert subagent_names() == ("explore",) == tuple(SUBAGENTS)


@pytest.mark.parametrize(
    ("arguments", "complaint"),
    [
        ({"prompt": "q"}, "agent must be a string"),
        ({"agent": 3, "prompt": "q"}, "agent must be a string, got int"),
        ({"agent": "explore"}, "prompt must be a string"),
        ({"agent": "explore", "prompt": None}, "prompt must be a string"),
        ({"agent": "explore", "prompt": "  \n"}, "prompt must not be empty"),
    ],
)
def test_arguments_that_are_not_an_agent_and_a_prompt_are_refused(ctx, arguments, complaint):
    with pytest.raises(ToolArgumentError, match=complaint):
        TaskTool().permission_request(ctx, arguments)


async def test_a_call_hands_the_agent_the_prompt_and_its_own_id_to_whatever_runs_it(ctx):
    spawn = Spawned(SubAgentResult("it is in a.py:3"))
    outcome = await TaskTool().run(
        replace(ctx, spawn=spawn), "call_9", {"agent": "explore", "prompt": "where is x?"}
    )

    assert spawn.calls == [SubAgentCall("explore", "where is x?", "call_9")]
    assert (outcome.tool_use_id, outcome.content, outcome.is_error) == (
        "call_9",
        "it is in a.py:3",
        False,
    )


async def test_an_error_from_the_sub_agent_is_an_error_result(ctx):
    spawn = Spawned(SubAgentResult("it could not finish", is_error=True))
    outcome = await TaskTool().run(
        replace(ctx, spawn=spawn), "c", {"agent": "explore", "prompt": "q"}
    )
    assert outcome.is_error and outcome.content == "it could not finish"


async def test_what_comes_back_is_stripped_of_terminal_controls_and_scrubbed_of_credentials(ctx):
    block, body = pem()
    text = "\x1b]0;title\x07done\n" + "\n".join(block)
    outcome = await TaskTool().run(
        replace(ctx, spawn=Spawned(SubAgentResult(text))), "c", {"agent": "explore", "prompt": "q"}
    )
    assert "\x1b" not in outcome.content and "\x07" not in outcome.content
    assert body[0] not in outcome.content and "[redacted:private-key]" in outcome.content


async def test_a_call_with_nothing_to_run_it_says_so_and_what_to_do_instead(
    ctx: ToolContext,
) -> None:
    assert ctx.spawn is None
    outcome = await TaskTool().run(ctx, "c", {"agent": "explore", "prompt": "q"})
    assert outcome.is_error
    assert "not available" in outcome.content and "yourself" in outcome.content


async def test_a_call_that_names_no_agent_is_an_error_even_when_it_is_run_directly(ctx):
    with pytest.raises(ToolArgumentError, match="no sub-agent named 'plan'"):
        await TaskTool().run(
            replace(ctx, spawn=Spawned(SubAgentResult("x"))), "c", {"agent": "plan", "prompt": "q"}
        )
