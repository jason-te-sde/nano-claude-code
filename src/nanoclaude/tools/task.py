# src/nanoclaude/tools/task.py
"""Task: hand a self-contained piece of work to a sub-agent and get its answer back.

The tool does no work of its own. It checks what the model asked for, and the session runs
the sub-agent (``agent/subagent.py``) through ``ToolContext.spawn``. What comes back is one
tool result: the sub-agent's final text, and nothing of how it got there.

An agent that does not exist is refused when the call is judged, before anybody is asked
whether to run it: a confirmation for a call that cannot succeed is a question that
teaches the person to stop reading them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanoclaude.permissions.policy import PermissionRequest
from nanoclaude.providers.base import ToolSpec
from nanoclaude.subagents import SUBAGENTS, SubAgent, SubAgentCall, subagent_names
from nanoclaude.tools.base import (
    ToolArgumentError,
    ToolContext,
    ToolOutcome,
    failed,
    ok,
    require_str,
    sanitize,
)

# Copied verbatim from spec 17.10. This is a prompt, not documentation.
DESCRIPTION = """Delegate a self-contained piece of work to a sub-agent.

The sub-agent starts with an empty conversation, runs its own tool calls, and returns
only its final answer. Use it when the work would otherwise fill your context with
detail you do not need to keep — searching a large codebase for where something is
defined, for example.

- `agent` selects which sub-agent to use; the user can list them with /agents.
- Give it a complete, self-contained prompt. It cannot see this conversation and cannot
  ask you follow-up questions.
- It cannot reach outside the working directory, and it cannot be granted permissions
  you do not have."""

SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "agent": {"type": "string", "description": "The sub-agent to run, by name."},
        "prompt": {"type": "string", "description": "What to do, complete in itself."},
    },
    "required": ["agent", "prompt"],
    "additionalProperties": False,
}


def _agent_named(arguments: Mapping[str, Any]) -> SubAgent:
    """The sub-agent a call names, or the error that lists the ones there are."""
    name = require_str(arguments, "agent")
    found = SUBAGENTS.get(name)
    if found is None:
        raise ToolArgumentError(
            f"no sub-agent named {name!r}; the available sub-agents are: "
            f"{', '.join(subagent_names())}"
        )
    return found


def _prompt_of(arguments: Mapping[str, Any]) -> str:
    prompt = require_str(arguments, "prompt")
    if not prompt.strip():
        raise ToolArgumentError("prompt must not be empty")
    return prompt


class TaskTool:
    name = "Task"
    # Not read-only, though no sub-agent in this version can change anything: a call starts
    # model requests that cost money, and it runs alone, one sub-agent at a time.
    read_only = False

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, DESCRIPTION, SCHEMA)

    def permission_request(
        self, _ctx: ToolContext, arguments: Mapping[str, Any]
    ) -> PermissionRequest:
        # The subject is the agent's name, so that a rule can be written for one of them.
        agent = _agent_named(arguments)
        _prompt_of(arguments)
        return PermissionRequest(self.name, agent.name, (), is_write=False)

    async def run(
        self, ctx: ToolContext, call_id: str, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        agent = _agent_named(arguments)
        prompt = _prompt_of(arguments)
        if ctx.spawn is None:
            return failed(
                call_id,
                "Task is not available here: this session has no way to start a sub-agent. "
                "Do the work yourself with the other tools.",
            )
        result = await ctx.spawn(SubAgentCall(agent.name, prompt, call_id))
        # The text of a model that read the project, on its way into the transcript of the one
        # that asked: stripped first and scrubbed after, as the executor does, since a secret
        # split by an escape sequence would otherwise get past the scrubber.
        text, _ = ctx.redactor.scrub(sanitize(result.text))
        return failed(call_id, text) if result.is_error else ok(call_id, text)
